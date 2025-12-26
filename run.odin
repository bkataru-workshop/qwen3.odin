/* Inference for GGUF Qwen-3 models in pure Odin */

package qwen3

import "core:fmt"
import "core:mem"
import "core:mem/virtual"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:testing"

// ----------------------------------------------------------------------------
// Transformer model
Config :: struct {
	dim:        uint, // transformer dimension
	hidden_dim: uint, // for ffn layers
	n_layers:   uint, // number of layers
	n_heads:    uint, // number of query heads
	n_kv_heads: uint, // numbery of key/value heads (can be < query heads because of multiquery)
	vocab_size: uint, // vocabulary size
	seq_len:    uint, // max sequence length
	head_dim:   uint, // attention dimension
}

Transformer_Weights :: struct {
	// token embedding table
	token_embedding_table: []f32, // (vocab_size, dim)
	// weights for rmsnorms in each layer
	rms_att_weight:        []f32, // (layer, dim)
	rms_ffn_weight:        []f32, // (layer, dim)
	// weights for matmuls
	wq:                    []f32, // (layer, dim, n_heads * head_dim)
	wk:                    []f32, // (layer, dim, n_kv_heads * head_dim)
	wv:                    []f32, // (layer, dim, n_kv_heads * head_dim)
	wo:                    []f32, // (layer, n_heads * head_dim, dim)
	wq_norm:               []f32, // (layer, head_dim)
	wk_norm:               []f32, // (layer, head_dim)
	// weights for ffn. w1 = up, w3 = gate, w2 = down
	w1:                    []f32, // (layer, dim, hidden_dim)
	w2:                    []f32, // (layer, hidden_dim, dim)
	w3:                    []f32, // (layer, dim, hidden_dim)
	// final rmsnorm
	rms_final_weight:      []f32, // (dim,)
	// Same as token_embedding_table. GGUF has the final layer anyway
	wcls:                  []f32,
}

Run_State :: struct {
	// current wave of activations
	x:           []f32, // activation at current time stamp (dim,)
	xb:          []f32, // buffer (dim,)
	xb2:         []f32, // an additional buffer just for convenience (dim,)
	xb3:         []f32, // an additional buffer just for convenience (att_head_dim,)
	hb:          []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
	hb2:         []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
	q:           []f32, // query (att_head_dim,)
	k:           []f32, // key (dim,)
	v:           []f32, // value (dim,)
	att:         []f32, // buffer for scores/attention values (n_heads, seq_len)
	logits:      []f32, // output logits
	// kv cache
	key_cache:   []f32, // (layer, seq_len, dim)
	value_cache: []f32, // (layer, seq_len, dim)
}

Transformer :: struct {
	config:    Config, // the hyperparameters of the architecture (the blueprint)
	weights:   Transformer_Weights, // the weights of the model
	state:     Run_State, // buffers for the "wave" of activations in the forward pass
	// TODO: remove the fd field? do we need this?
	// fd:        int, // file descriptor for memory mapping
	data:      []f32, // memory mapped data pointer
	file_size: int, // size of the checkpoint file in bytes
}

malloc_run_state :: proc(s: ^Run_State, p: Config) {
	att_head_dim := p.n_heads * p.head_dim
	kv_dim := p.n_kv_heads * p.head_dim // 1024

	err: mem.Allocator_Error

	s.x, err = make([]f32, p.dim)
	s.xb, err = make([]f32, p.dim)
	s.xb2, err = make([]f32, p.dim)
	s.xb3, err = make([]f32, att_head_dim)
	s.hb, err = make([]f32, p.hidden_dim)
	s.hb2, err = make([]f32, p.hidden_dim)
	s.q, err = make([]f32, att_head_dim)
	s.k, err = make([]f32, kv_dim)
	s.v, err = make([]f32, kv_dim)
	s.att, err = make([]f32, p.n_heads * p.seq_len)
	s.logits, err = make([]f32, p.vocab_size)
	s.key_cache, err = make([]f32, p.n_layers * p.seq_len * kv_dim)
	s.value_cache, err = make([]f32, p.n_layers * p.seq_len * kv_dim)

	// ensure all mallocs went fine
	if err != .None {
		fmt.eprintf("malloc failed: %v\n", err)
		os.exit(1)
	}
}

free_run_state :: proc(s: ^Run_State) {
	delete(s.x)
	delete(s.xb)
	delete(s.xb2)
	delete(s.xb3)
	delete(s.hb)
	delete(s.hb2)
	delete(s.q)
	delete(s.k)
	delete(s.v)
	delete(s.att)
	delete(s.logits)
	delete(s.key_cache)
	delete(s.value_cache)
}

bytes_as_floats :: proc(data: []u8) -> []f32 {
	// 1. Check if the length is a multiple of size_of(f32)
	if len(data) % size_of(f32) != 0 {
		fmt.eprintln("Byte slice length is not a multiple of 4")
		os.exit(1)
	}

	// 2. Check if the data pointer is 4-byte aligned
	// uintptr is used for pointer arithmetic/alignment checks
	if uintptr(raw_data(data)) % align_of(f32) != 0 {
		fmt.eprintln("Data is not 4-byte aligned")
		os.exit(1)
	}

	// 3. Perform the cast
	ptr := ([^]f32)(raw_data(data))
	new_slice := ptr[:len(data) / 4]

	// could have also done this:
	// mem.slice_ptr converts a raw pointer and length into a slice
	// new_slice := mem.slice_ptr(ptr, len(data) / 4)

	return new_slice
}


// Map GGUF layers to transformer weights
memory_map_weights :: proc(w: ^Transformer_Weights, p: Config, data: []f32) {
	offset: uint = 0

	w.wcls = data[offset:offset + p.vocab_size * p.dim] // last layer in TR
	offset += p.vocab_size * p.dim
	w.rms_final_weight = data[offset:offset + p.dim] // right before the last
	offset += p.dim
	w.token_embedding_table = data[offset:offset + p.vocab_size * p.dim] // first layer
	offset += p.vocab_size * p.dim
	w.wk = data[offset:offset + p.dim * (p.n_kv_heads * p.head_dim)]
	offset += p.dim * (p.n_kv_heads * p.head_dim) // 1024 x 1024 = dim (1024) x num_kv_heads (8) x p->head_dim (128)
	w.wk_norm = data[offset:offset + p.head_dim]
	offset += p.head_dim // head_dim (128)
	w.rms_att_weight = data[offset:offset + p.dim]
	offset += p.dim // dimension (1024)
	w.wo = data[offset:offset + (p.n_heads * p.head_dim) * p.dim]
	offset += (p.n_heads * p.head_dim) * p.dim // attention heads (16) x head dim (128) * dim
	w.wq = data[offset:offset + p.dim * (p.n_heads * p.head_dim)]
	offset += p.dim * (p.n_heads * p.head_dim)
	w.wq_norm = data[offset:offset + p.head_dim]
	offset += p.head_dim // head_dim (128)
	w.wv = data[offset:offset + p.dim * (p.n_kv_heads * p.head_dim)]
	offset += p.dim * (p.n_kv_heads * p.head_dim) // equal to wk
	w.w2 = data[offset:offset + p.hidden_dim * p.dim]
	offset += p.hidden_dim * p.dim // ffn.down 3072 * 1024
	w.w3 = data[offset:offset + p.dim * p.hidden_dim]
	offset += p.dim * p.hidden_dim // ffn.gate
	w.rms_ffn_weight = data[offset:offset + p.dim]
	offset += p.dim // ffn.norm
	w.w1 = data[offset:offset + p.dim * p.hidden_dim]
	offset += p.dim * p.hidden_dim // ffn.up
}

// --------------------------------------
// read GGUF
read_checkpoint :: proc(
	checkpoint_path: string,
	config: Config,
	weights: ^Transformer_Weights,
	data: ^[]f32,
	file_size: ^int,
) {
	mmap, err := virtual.map_file_from_path(checkpoint_path, {.Read})
	if err != .None {
		fmt.eprintf("mmap failed: %v\n", err)
		os.exit(1)
	}
	header_offset: uint = 5951648

	fmt.printf("file size is %d\n", len(mmap))
	file_size^ = len(mmap)

	data^ = bytes_as_floats(mmap[header_offset:]) // skip header bytes. header_size = 5951648 TODO
	// gguf total header = file size - (last tensor size + last offset)

	memory_map_weights(weights, config, data^)
}

build_transformer :: proc(t: ^Transformer, checkpoint_path: string) {
	// read in the Weights from the GGUF
	read_checkpoint(checkpoint_path, t.config, &t.weights, &t.data, &t.file_size)
	// allocate the Run_State buffers
	malloc_run_state(&t.state, t.config)
}

free_transformer :: proc(t: ^Transformer) {
	if t.data != nil {
		virtual.release(raw_data(t.data), len(t.data))
		t.data = nil
	}
}

// load the GGUF config file
load_config :: proc(t: ^Transformer, filename: string = "header.txt") {
	data, ok := os.read_entire_file(filename)
	if !ok {
		fmt.eprintf("Failed to open %s\n", filename)
		os.exit(1)
	}
	defer delete(data)

	oks: [8]bool

	it := string(data)
	for line in strings.split_lines_iterator(&it) {
		cleaned := strings.trim_space(line)

		if strings.index(cleaned, "=") == -1 do continue

		parts := strings.split_n(cleaned, "=", 2)
		defer delete(parts)

		key, value := parts[0], parts[1]

		if (len(key) == 0 || len(value) == 0) do continue

		switch key {
		case "QWEN3_EMBEDDING_LENGTH":
			t.config.dim, oks[0] = strconv.parse_uint(value, 10)
		case "QWEN3_FEED_FORWARD_LENGTH":
			t.config.hidden_dim, oks[1] = strconv.parse_uint(value, 10)
		case "QWEN3_BLOCK_COUNT":
			t.config.n_layers, oks[2] = strconv.parse_uint(value, 10)
		case "QWEN3_ATTENTION_HEAD_COUNT":
			t.config.n_heads, oks[3] = strconv.parse_uint(value, 10)
		case "QWEN3_ATTENTION_HEAD_COUNT_KV":
			t.config.n_kv_heads, oks[4] = strconv.parse_uint(value, 10)
		case "QWEN3_CONTEXT_LENGTH":
			t.config.seq_len, oks[5] = strconv.parse_uint(value, 10)
		case "QWEN3_ATTENTION_KEY_LENGTH":
			t.config.head_dim, oks[6] = strconv.parse_uint(value, 10)
		case "TOKENIZER_GGML_TOKENS":
			ARRAY_LENGTH_KEY :: "ARRAY_LENGTH="

			if needle := strings.index(value, ARRAY_LENGTH_KEY); needle != -1 {
				start := needle + len(ARRAY_LENGTH_KEY)
				subvalue := value[start:]
				t.config.vocab_size, oks[7] = strconv.parse_uint(subvalue, 10)
			} else {
				fmt.eprintf("No key named '%s' found in config", ARRAY_LENGTH_KEY)
				os.exit(1)
			}
		}
	}

	for ok in oks {
		if !ok {
			fmt.eprintln("Invalid or corrupted config, didn't find exactly eight keys")
			os.exit(1)
		}
	}
}

@(test)
config_load :: proc(t: ^testing.T) {
	tr: Transformer

	load_config(&tr, "header.txt")

	testing.expect_value(t, tr.config.dim, 1024)
	testing.expect_value(t, tr.config.hidden_dim, 3072)
	testing.expect_value(t, tr.config.n_layers, 28)
	testing.expect_value(t, tr.config.n_heads, 16)
	testing.expect_value(t, tr.config.n_kv_heads, 8)
	testing.expect_value(t, tr.config.vocab_size, 151936)
	testing.expect_value(t, tr.config.seq_len, 40960)
	testing.expect_value(t, tr.config.head_dim, 128)
}
