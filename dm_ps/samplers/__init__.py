"""Load only the selected sampler and its optional dependencies."""

from importlib import import_module


def _lazy_sampler(module, function):
    def sample(*args, **kwargs):
        return getattr(import_module(module), function)(*args, **kwargs)
    sample.__name__ = function
    return sample


dps = _lazy_sampler('dm_ps.samplers.dps_ve', 'dps')
mgdm = _lazy_sampler('dm_ps.samplers.mgdm_ve', 'mgdm')
replica_algo = _lazy_sampler('dm_ps.samplers.replica_exhange.algo', 'replica_algo')
daps = _lazy_sampler('dm_ps.samplers.daps', 'daps')
pnp_flow = _lazy_sampler('dm_ps.samplers.pnp_flow', 'pnp_flow')
reddiff = _lazy_sampler('dm_ps.samplers.reddiff', 'reddiff')
tds = _lazy_sampler('dm_ps.samplers.tds', 'tds')
mgps = _lazy_sampler('dm_ps.samplers.mgps_ve', 'mgps')
idw_map = _lazy_sampler('dm_ps.samplers.rain_baselines', 'idw_map')
gmz_map = _lazy_sampler('dm_ps.samplers.rain_baselines', 'gmz_map')
ok_map = _lazy_sampler('dm_ps.samplers.rain_baselines', 'ok_map')


AVAILABLE_SAMPLERS = {
    # diffusion samplers
    "mgdm": mgdm,
    "dps": dps,
    "daps": daps,
    "crepe": replica_algo,
    "pnp_flow": pnp_flow,
    "reddiff": reddiff,
    "tds": tds,
    "mgps": mgps,
    # rain baselines
    "idw": idw_map,
    "gmz": gmz_map,
    "ok": ok_map,
}
