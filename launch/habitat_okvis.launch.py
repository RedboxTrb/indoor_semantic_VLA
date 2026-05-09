from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

_TORCH_LIB = os.path.expanduser('~/libtorch_cu126/libtorch/lib')
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':' + _TORCH_LIB


def generate_launch_description():
    okvis_pkg = get_package_share_directory('okvis')

    okvis_config = os.path.expanduser('~/habitat_slam/habitat_okvis2.yaml')
    se_config    = os.path.expanduser('~/habitat_slam/se2.yaml')
    mesh_file    = os.path.join(okvis_pkg, 'resources', 'meshes', 'realsense.dae')

    csv_path_arg = DeclareLaunchArgument('csv_path', default_value='/tmp/habitat_okvis')

    torch_lib_env = SetEnvironmentVariable(
        'LD_LIBRARY_PATH',
        [os.environ.get('LD_LIBRARY_PATH', ''), ':', _TORCH_LIB]
    )
    malloc_env = SetEnvironmentVariable('MALLOC_CHECK_', '0')

    okvis_node = Node(
        package='okvis',
        executable='okvis2x_language_network_node_subscriber',
        name='okvis',
        namespace='okvis',
        output='screen',
        parameters=[{
            'config_filename':    okvis_config,
            'se_config_filename': se_config,
            'csv_path':           LaunchConfiguration('csv_path'),
            # mesh_cutoff_z removed — was clipping visualization in world-Z direction
            'save_submap_meshes': True,
            'mesh_file':          f'file://{mesh_file}',
        }],
        remappings=[
            # cam0: left stereo (gray, SLAM only)
            ('/okvis/cam0/image_raw',        '/d435i_depth_camera/image_raw'),
            ('/okvis/cam0/camera_info',      '/d435i_depth_camera/camera_info'),
            # cam1: right stereo (gray, SLAM only)
            ('/okvis/cam1/image_raw',        '/d435i_depth_camera/right/image_raw'),
            ('/okvis/cam1/camera_info',      '/d435i_depth_camera/right/camera_info'),
            # cam2: RGB+depth mapping camera (co-located with cam0, same topics)
            # rgb — VLProcessor runs eSAM+CLIP on this
            ('/okvis/cam2/image_raw',        '/d435i_depth_camera/image_raw'),
            ('/okvis/cam2/camera_info',      '/d435i_depth_camera/camera_info'),
            # depth2 — depth co-registered with cam2 RGB for correct 3D integration
            ('/okvis/depth2/image_raw',      '/d435i_depth_camera/depth/image_raw'),
            ('/okvis/depth2/camera_info',    '/d435i_depth_camera/depth/camera_info'),
            ('/okvis/imu0',                  '/imu/data'),
        ],
    )

    return LaunchDescription([
        torch_lib_env,
        malloc_env,
        csv_path_arg,
        okvis_node,
    ])
