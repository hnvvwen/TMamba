from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/got10k_lmdb'
    settings.got10k_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/itb'
    settings.lasot_extension_subset_path_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/lasot_lmdb'
    settings.lasot_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/lasot'
    settings.network_path = '/home/admz/projects/pythonprojects/TrackingMamba/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/nfs'
    settings.otb_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/otb'
    settings.prj_dir = '/root/TrackingMamba'
    settings.result_plot_path = '/home/admz/projects/pythonprojects/TrackingMamba/output/test/result_plots'
    settings.results_path = '/home/admz/projects/pythonprojects/TrackingMamba/output/test/tracking_results'    # Where to store tracking results
    #settings.save_dir = '/home/admz/projects/pythonprojects/TrackingMamba/output'
    settings.save_dir = '/root/TrackingMamba/output'
    settings.segmentation_path = '/home/admz/projects/pythonprojects/TrackingMamba/output/test/segmentation_results'
    settings.tc128_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/trackingnet'

    settings.vot18_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/vot2018'
    settings.vot22_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/vot2022'
    settings.vot_path = '/home/admz/projects/pythonprojects/TrackingMamba/data/VOT2019'
    settings.uav_path = '/home/admz/projects/datasets/UAV123'
    settings.dtb70_path = '/home/admz/projects/datasets/DTB70'
    settings.odinmj_path = '/data/OTMJ'
    settings.youtubevos_dir = ''

    return settings

