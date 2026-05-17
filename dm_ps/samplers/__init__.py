from dm_ps.samplers.dps_ve import dps
from dm_ps.samplers.mgdm_ve import mgdm
from dm_ps.samplers.replica_exhange.algo import replica_algo
from dm_ps.samplers.daps import daps
from dm_ps.samplers.pnp_flow import pnp_flow
from dm_ps.samplers.reddiff import reddiff
from dm_ps.samplers.tds import tds
from dm_ps.samplers.mgps_ve import mgps

from dm_ps.samplers.rain_baselines import idw_map, gmz_map, ok_map


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
