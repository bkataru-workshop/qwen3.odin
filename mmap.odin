package qwen3

import "core:fmt"
import "core:mem/virtual"
import "core:os"

mmap_main :: proc() {
	filename := "mmap_demo.bin"
	initial_content := "Odin MMap Test Data"

	if !os.write_entire_file(filename, transmute([]u8)initial_content) {
		fmt.println("Error: Could not create test file.")
		return
	}
	defer os.remove(filename)

	data, err := virtual.map_file_from_path(filename, {.Read, .Write})
	if err != .None {
		fmt.printf("Memory mapping failed: %v\n", err)
		return
	}

	defer virtual.release(raw_data(data), len(data))

	fmt.printf("Original content: %s\n", data)

	if len(data) >= 4 {
		data[0] = 'H'
		data[1] = 'E'
		data[2] = 'L'
		data[3] = 'P'
	}
	fmt.printf("Modified in-memory: %s\n", data)

	fmt.println("Data has been modified through the memory-mapped slice.")


}
