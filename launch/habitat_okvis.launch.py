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
    se_config    = os.path.join(okvis_pkg, 'config', 'rsD455', 'se2.yaml')
    mesh_file    = os.path.join(okvis_pkg, 'resources', 'meshes', 'realsense.dae')

    csv_path_arg = DeclareLaunchArgument('csv_path', default_value='/tmp/habitat_okvis')

    torch_lib_env = SetEnvironmentVariable(
        'LD_LIBRARY_PATH',
        [os.environ.get('LD_LIBRARY_PATH', ''), ':', _TORCH_LIB]
    )

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
            'mesh_cutoff_z':      2.5,
            'save_submap_meshes': False,
            'mesh_file':          f'file://{mesh_file}',
        }],
        remappings=[
            ('/okvis/cam0/image_raw',        '/d435i_depth_camera/image_raw'),
            ('/okvis/cam0/camera_info',      '/d435i_depth_camera/camera_info'),
            ('/okvis/cam1/image_raw',        '/d435i_depth_camera/image_raw'),
            ('/okvis/cam1/camera_info',      '/d435i_depth_camera/camera_info'),
            ('/okvis/depth0/image_raw',      '/d435i_depth_camera/depth/image_raw'),
            ('/okvis/depth0/camera_info',    '/d435i_depth_camera/depth/camera_info'),
            ('/okvis/cam2/image_raw',        '/d435i_depth_camera/image_raw'),
            ('/okvis/cam2/camera_info',      '/d435i_depth_camera/camera_info'),
            ('/okvis/imu0',                  '/imu/data'),
        ],
    )

    return LaunchDescription([
        torch_lib_env,
        csv_path_arg,
        okvis_node,
    ])
