# SeqMM: A Benchmark Suite for Evaluating Matrix Multiplications in Sequence Processing and Generation

## Project Structure

### Benchmarks

`cublas_example`, `cusparse_example` and `paisparse_example` are used to benchmark CuBLAS, CuSparse and PaiSparse implementations with random inputs.

`plug_example` is used for benchmarking with real weights data aquired from the PLUG model.
The steps for the PLUG weights aquiring is illustrated in the "Scripts" section.

### Tests

### Scripts

Fill `parser_config.conf` and run `model_parser.py` to aquire model weights as well as their shapes.

## Build

```bash
~> mkdir build && cd build
~> cmake ../ -DCUDA_ARCHS="60;61;70;75"
~> make -j4
```