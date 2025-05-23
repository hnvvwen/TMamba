from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/root/TMamba/data/got10k_lmdb'
    settings.got10k_path = '/root/TMamba/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/root/TMamba/data/itb'
    settings.lasot_extension_subset_path_path = '/root/TMamba/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/root/TMamba/data/lasot_lmdb'
    settings.lasot_path = '/root/TMamba/data/lasot'
    settings.network_path = '/root/TMamba/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/root/TMamba/data/nfs'
    settings.otb_path = '/root/TMamba/data/otb'
    settings.prj_dir = '/root/TMamba'
    settings.result_plot_path = '/root/TMamba/output/test/result_plots'
    settings.results_path = '/root/TMamba/output/test/tracking_results'    # Where to store tracking results
    #settings.save_dir = '/home/admz/projects/pythonprojects/TrackingMamba/output'
    settings.save_dir = '/root/TMamba/output'
    settings.segmentation_path = '/root/TMamba/output/test/segmentation_results'
    settings.tc128_path = '/root/TMamba/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/root/TMamba/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/root/TMamba/data/trackingnet'

    settings.vot18_path = '/root/TMamba/data/vot2018'
    settings.vot22_path = '/root/TMamba/data/vot2022'
    settings.vot_path = '/root/TMamba/data/VOT2019'
    settings.uav_path = '/home/admz/projects/datasets/UAV123'
    settings.dtb70_path = '/home/admz/projects/datasets/DTB70'
    settings.odinmj_path = '/data/OTMJ'
    settings.youtubevos_dir = ''

    return settings

