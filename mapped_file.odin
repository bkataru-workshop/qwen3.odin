package qwen3

import "core:fmt"
import "core:mem/virtual"
import "core:os"

Mapped_File :: struct {
	data: []u8,
}

// map_file encapsulates map_file_from_path
map_file :: proc(
	path: string,
	flags: virtual.Map_File_Flags = {.Read, .Write},
) -> (
	mf: Mapped_File,
	err: virtual.Map_File_Error,
) {
	mf.data, err = virtual.map_file_from_path(path, flags)
	return
}

unmap :: proc(mf: ^Mapped_File) {
	if mf.data != nil {
		virtual.release(raw_data(mf.data), len(mf.data))
		mf.data = nil
	}
}

main :: proc() {
	filename := "mmap_demo.bin"
	os.write_entire_file(filename, transmute([]u8)string("Odin MMap Test Data"))
	defer os.remove(filename)

	mfile, err := map_file(filename)
	if err != .None {
		fmt.printf("Error mapping file: %v\n", err)
		return
	}
	defer unmap(&mfile)

	fmt.printf("Mapped data: %s\n", mfile.data)
}
