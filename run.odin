package qwen3

import "core:fmt"
import "core:mem"
import "core:mem/virtual"
import "core:os"

/* Inference for GGUF Qwen-3 models in pure Odin */

// ----------------------------------------------------------------------------
// Transformer model
Config :: struct {
	dim:        int, // transformer dimension
	hidden_dim: int, // for ffn layers
	n_layers:   int, // number of layers
	n_heads:    int, // number of query heads
	n_kv_heads: int, // numbery of key/value heads (can be < query heads because of multiquery)
	vocab_size: int, // vocabulary size
	seq_len:    int, // max sequence length
	head_dim:   int, // attention dimension
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
	fd:        int, // file descriptor for memory mapping
	data:      []f32, // memory mapped data pointer
	file_size: int, // size of the checkpoint file in bytes
}

malloc_run_state :: proc(config: Config) -> (state: Run_State) {
	att_head_dim := config.n_heads * config.head_dim
	kv_dim := config.n_kv_heads * config.head_dim // 1024

	err: mem.Allocator_Error

	state.x, err = make([]f32, config.dim)
	state.xb, err = make([]f32, config.dim)
	state.xb2, err = make([]f32, config.dim)
	state.xb3, err = make([]f32, att_head_dim)
	state.hb, err = make([]f32, config.hidden_dim)
	state.hb2, err = make([]f32, config.hidden_dim)
	state.q, err = make([]f32, att_head_dim)
	state.k, err = make([]f32, kv_dim)
	state.v, err = make([]f32, kv_dim)
	state.att, err = make([]f32, config.n_heads * config.seq_len)
	state.logits, err = make([]f32, config.n_layers * config.seq_len * kv_dim)
	state.key_cache, err = make([]f32, config.n_layers * config.seq_len * kv_dim)
	state.value_cache, err = make([]f32, config.n_layers * config.seq_len * kv_dim)

	// ensure all mallocs went fine
	if err != .None {
		fmt.eprintf("malloc failed: %v\n", err)
		os.exit(1)
	}

	return state
}

free_run_state :: proc(state: ^Run_State) {
	delete(state.x)
	delete(state.xb)
	delete(state.xb2)
	delete(state.xb3)
	delete(state.hb)
	delete(state.hb2)
	delete(state.q)
	delete(state.k)
	delete(state.v)
	delete(state.att)
	delete(state.logits)
	delete(state.key_cache)
	delete(state.value_cache)
}

Byte_Slice_Error :: enum {
	None,
	Length_Not_Multiple_Of_4,
	Data_Not_4_Byte_Aligned,
}

bytes_as_floats :: proc(data: []u8) -> ([]f32, Byte_Slice_Error) {
	// 1. Check if the length is a multiple of size_of(f32)
	if len(data) % size_of(f32) != 0 {
		return nil, .Length_Not_Multiple_Of_4
	}

	// 2. Check if the data pointer is 4-byte aligned
	// uintptr is used for pointer arithmetic/alignment checks
	if uintptr(raw_data(data)) % align_of(f32) != 0 {
		return nil, .Data_Not_4_Byte_Aligned
	}

	// 3. Perform the cast
	ptr := ([^]f32)(raw_data(data))
	new_slice := ptr[:len(data) / 4]

	// could have also done this:
	// mem.slice_ptr converts a raw pointer and length into a slice
	// new_slice := mem.slice_ptr(ptr, len(data) / 4)

	return new_slice, .None
}


// Map GGUF layers to transformer weights
memory_map_weights :: proc(
	data: []u8,
	config: Config,
	header_offset: uint,
) -> (
	weights: Transformer_Weights,
) {
	float_data, err := bytes_as_floats(data[header_offset:])
	switch err {
	case .Length_Not_Multiple_Of_4:
		fmt.eprintln("Byte slice length is not a multiple of 4")
		os.exit(1)
	case .Data_Not_4_Byte_Aligned:
		fmt.eprintln("Data is not 4-byte aligned")
		os.exit(1)
	case .None:
	}
	offset := 0

	weights.wcls = float_data[offset:offset + config.vocab_size * config.dim] // last layer in TR
	offset += config.vocab_size * config.dim
	weights.rms_final_weight = float_data[offset:offset + config.dim] // right before the last
	offset += config.dim
	weights.token_embedding_table = float_data[offset:offset + config.vocab_size * config.dim] // first layer
	offset += config.vocab_size * config.dim
	weights.wk = float_data[offset:offset + config.dim * (config.n_kv_heads * config.head_dim)]
	offset += config.dim * (config.n_kv_heads * config.head_dim) // 1024 x 1024 = dim (1024) x num_kv_heads (8) x p->head_dim (128)
	weights.wk_norm = float_data[offset:offset + config.head_dim]
	offset += config.head_dim // head_dim (128)
	weights.rms_att_weight = float_data[offset:offset + config.dim]
	offset += config.dim // dimension (1024)
	weights.wo = float_data[offset:offset + (config.n_heads * config.head_dim) * config.dim]
	offset += (config.n_heads * config.head_dim) * config.dim // attention heads (16) x head dim (128) * dim
	weights.wq = float_data[offset:offset + config.dim * (config.n_heads * config.head_dim)]
	offset += config.dim * (config.n_heads * config.head_dim)
	weights.wq_norm = float_data[offset:offset + config.head_dim]
	offset += config.head_dim // head_dim (128)
	weights.wv = float_data[offset:offset + config.dim * (config.n_kv_heads * config.head_dim)]
	offset += config.dim * (config.n_kv_heads * config.head_dim) // equal to wk
	weights.w2 = float_data[offset:offset + config.hidden_dim * config.dim]
	offset += config.hidden_dim * config.dim // ffn.down 3072 * 1024
	weights.w3 = float_data[offset:offset + config.dim * config.hidden_dim]
	offset += config.dim * config.hidden_dim // ffn.gate
	weights.rms_ffn_weight = float_data[offset:offset + config.dim]
	offset += config.dim // ffn.norm
	weights.w1 = float_data[offset:offset + config.dim * config.hidden_dim]
	offset += config.dim * config.hidden_dim // ffn.up

	return weights
}

// --------------------------------------
// read GGUF
read_checkpoint :: proc(path: string, config: Config) -> Transformer {
	mmap, err := virtual.map_file_from_path(path, {.Read})
	if err != .None {
		fmt.eprintf("mmap failed: %v\n", err)
		os.exit(1)
	}
	header_offset := 5951648

	fmt.printf("file size is %d\n", len(mmap))

	weights := memory_map_weights(mmap, config, header_offset)
	state, err := malloc_run_state(config)
	if err != .None {
		fmt.println("malloc failed")
	}

}
