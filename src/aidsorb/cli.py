import fire
from . visualize import draw_pcd_from_file
from . utils import pcd_from_dir
from . data import prepare_data


def aidsorb_fire():
    fire.Fire({
        'visualize': draw_pcd_from_file,
        'create': pcd_from_dir,
        'prepare': prepare_data
        })
