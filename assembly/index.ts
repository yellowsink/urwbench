@external("env", "Deno.readFileSync")
declare function denoReadFileSync(s: string): Uint8Array;

export function downscaleBasic(buf: Float32Array, size: i32): Float32Array {
  const fchunkSz = Math.ceil(<f64>buf.length / <f64>size);
  const nChunks = <i32>Math.ceil(<f64>buf.length / fchunkSz);
  const chunkSz = <i32>fchunkSz;

  const res = new Float32Array(nChunks);
  for (let ci = 0; ci < nChunks; ci++) {
    let max: f64 = 0;
    const chunk = buf.subarray(ci * chunkSz, (ci + 1) * chunkSz);

    for (let i = 1; i < chunk.length; i++) {
      max = Math.max(chunk[i], max);
    }

    res[ci] = <f32>max;
  }

  return res;
}

export function downscaleSimdLinear(buf: Float32Array, size: i32): Float32Array {
  // round chunk size up to a multiple of 4
  const fchunkSz = 4 * Math.ceil((<f64>buf.length / <f64>size) / 4);
  const nChunks = <i32>Math.ceil(<f64>buf.length / fchunkSz);
  const chunkSz = <i32>fchunkSz;

  const res = new Float32Array(nChunks);
  for (let ci = 0; ci < nChunks; ci++) {
    let max = v128.splat<f32>(0);
    const chunk = buf.subarray(ci * chunkSz, (ci + 1) * chunkSz);
    const ptr = chunk.dataStart;

    // 4 floats per vec
    for (let i = 1; i < chunk.length; i += 4) {
      const vec = v128.load(ptr + i * sizeof<v128>())
      max = f32x4.max(max, vec);
    }

    res[ci] = Mathf.max(
      Mathf.max(f32x4.extract_lane(max, 0), f32x4.extract_lane(max, 1)),
      Mathf.max(f32x4.extract_lane(max, 2), f32x4.extract_lane(max, 3)),
    );
  }

  return res;
}

export function downscaleSimdAcc(buf: Float32Array, size: i32): Float32Array {
  // round chunk size up to a multiple of 4
  const fchunkSz = 4 * Math.ceil((<f64>buf.length / <f64>size) / 4);
  const nChunks = <i32>Math.ceil(<f64>buf.length / fchunkSz);
  const chunkSz = <i32>fchunkSz;

  const res = new Float32Array(nChunks);
  for (let ci = 0; ci < nChunks; ci++) {
    let max = v128.splat<f32>(0);
    const chunk = buf.subarray(ci * chunkSz, (ci + 1) * chunkSz);
    const ptr = chunk.dataStart;

    // 4 floats per vec
    for (let i = 1; i < chunk.length; i += 4) {
      const vec = v128.load(ptr + i * sizeof<v128>())
      max = f32x4.max(max, vec);
    }

    // [a; b; c; d] max with
    // [c; d; x, x]
    // gives [max(a,c), max(b,d); x; x]
    // so can change three max ops to one simd max and one max
    const maxed2 = f32x4.max(
        max,
        v128.swizzle(
            max,
            v128(8, 9, 10, 11, // lane 3
            12, 13, 14, 15, // lane 4
            0, 1, 2, 3, 0, 1, 2, 3) // we don't care about the upper half, just use lane 1
        )
        );
    // lane 1 is max(a,c), lane 2 is max(b,d)
    res[ci] = Mathf.max(f32x4.extract_lane(max, 0), f32x4.extract_lane(max, 1));
  }

  return res;
}

export function benchDownscale(): void {
  const N = 100;
  const DOWNSCALE_SIZE = 1000;
  const rawbuf = denoReadFileSync("Corsair.dat").buffer;
  const DATA = Float32Array.wrap(rawbuf, 0, rawbuf.byteLength / Float32Array.BYTES_PER_ELEMENT);

  console.log("START BENCHMARK");

  // compute cost of looping
  let start = performance.now();
  for (let i = 0; i < N; i++) {}
  const loopTime = performance.now() - start;

  // test
  start = performance.now();
  for (let i = 0; i < N; i++)
    downscaleBasic(DATA, DOWNSCALE_SIZE);
  const basicTime = performance.now() - start;

  start = performance.now();
  for (let i = 0; i < N; i++)
    downscaleSimdLinear(DATA, DOWNSCALE_SIZE);
  const simdLinearTime = performance.now() - start;

  start = performance.now();
  for (let i = 0; i < N; i++)
    downscaleSimdAcc(DATA, DOWNSCALE_SIZE);
  const simdAccTime = performance.now() - start;

  console.log("basic time:       " + ((basicTime - loopTime) / N).toString().padStart(5) + " ms");
  console.log("simd linear time: " + ((simdLinearTime - loopTime) / N).toString().padStart(5) + " ms");
  console.log("simd acc time:    " + ((simdLinearTime - loopTime) / N).toString().padStart(5) + " ms");

}

//const fchunkSz = Math.pow(4, Math.ceil(Math.log2(Math.ceil(<f64>buf.length / <f64>size)) / 2));