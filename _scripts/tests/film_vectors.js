// Reference numbers for gyroflow-core's film port, computed by Photo360's film.js itself.
// Usage: node _scripts/tests/film_vectors.js <path to Photo360/film.js> > src/core/film/test_vectors.json
'use strict';
const Film = require(process.argv[2] || 'C:/Users/Leo/Gits/Photo360/film.js');
const u = (v) => v >>> 0;
const pcgIn = [0, 1, 2, 3, 0x51, 123456789, 0x7fffffff, 0x80000000 | 0, -1, 0x3C6EF372];
const hashIn = [[0, 0, 0, 0], [1, 2, 3, 4], [-5, 7, 0x51, 0x3C6EF372], [4000, 3000, 12345, 0x3C6EF372]];
const golden = Film.STOCKS.find((s) => s.id === 'golden200');
const tungsten = Film.STOCKS.find((s) => s.id === 'tungsten800');
const model = (st, s) => Film.grainModel(st.grain.radiusMedianPx, st.grain.radiusLogSigma, st.grain.filterSigmaPx).forRenderScale(s);
const setup = (st, s, z) => { const g = Film.grainSetup(st.grain, s, z); return { sigma: g.sigma, radius: g.radius, norm: g.norm, std50: g.std[50], std100: g.std[100], std180: g.std[180] }; };
const out = {
  pcg: pcgIn.map((v) => [u(v), u(Film.pcg(v))]),
  hash4: hashIn.map((a) => [a.map(u), u(Film.hash4(...a))]),
  gaussian: [0, 1, 99, 0x3C6EF372, 0xdeadbeef | 0].map((h) => [u(h), Film.gaussian(h)]),
  fnv: ['', 'C0001.MP4', 'clip ä.mov'].map((s) => [s, u(Film.seedOf(s))]),
  layer_seeds: [[u(Film.seedOf('C0001.MP4')), Film.layerSeeds(Film.seedOf('C0001.MP4')).map(u)]],
  inverse_normal: [0.001, 0.02, 0.3, 0.5, 0.9, 0.99].map((p) => [p, Film.inverseNormal(p)]),
  i0e: [0, 0.5, 3.75, 4, 20].map((z) => [z, Film.i0e(z)]),
  model: [[golden, 1], [golden, 0.278], [tungsten, 1.667]].map(([st, s]) => {
    const m = model(st, s);
    return { id: st.id, scale: s, var10: m.filteredVariance(0.1), var50: m.filteredVariance(0.5), var90: m.filteredVariance(0.9), lag1: m.lagOneCorrelation() };
  }),
  kernel_lag_one: [0.5, 1, 2.5].map((s) => [s, Film.kernelLagOne(s)]),
  setup: [
    { id: 'golden200', scale: Film.quantizeScale(0.5), z: 1, ...setup(golden, Film.quantizeScale(0.5), 1) },
    { id: 'golden200', scale: Film.quantizeScale(0.5), z: Film.quantizeScale(0.3), ...setup(golden, Film.quantizeScale(0.5), Film.quantizeScale(0.3)) },
    { id: 'tungsten800', scale: Film.quantizeScale(1.2), z: 1, ...setup(tungsten, Film.quantizeScale(1.2), 1) },
  ],
  quantize: [0.1, 0.278, 0.5, 1, 1.667].map((v) => [v, Film.quantizeScale(v)]),
  halo_factor: [3, 8, 9, 41, 244, 2000].map((r) => [r, Film.haloFactor(r)]),
};
process.stdout.write(JSON.stringify(out, null, 1) + '\n');
