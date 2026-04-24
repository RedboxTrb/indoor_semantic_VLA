from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import xacro


def generate_launch_description():
    pkg      = get_package_share_directory('basebot_urdf_description')
    nav2_pkg = get_package_share_directory('nav2_bringup')

    xacro_file        = os.path.join(pkg, 'urdf', 'labbot.xacro')
    robot_description = xacro.process_file(xacro_file).toxml()
    default_world     = os.path.join(pkg, 'worlds', 'envs.world')
    slam_params       = os.path.join(pkg, 'config', 'slam_params.yaml')
    nav2_params       = os.path.join(pkg, 'config', 'nav2_params.yaml')

    world_arg = DeclareLaunchArgument('world',     default_value=default_world)
    spawn_x   = DeclareLaunchArgument('spawn_x',   default_value='0.0')
    spawn_y   = DeclareLaunchArgument('spawn_y',   default_value='0.0')
    spawn_z   = DeclareLaunchArgument('spawn_z',   default_value='0.1')
    spawn_yaw = DeclareLaunchArgument('spawn_yaw', default_value='0.0')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'),
                         'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': LaunchConfiguration('world')}.items()
    )

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}]
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'labbot', '-topic', 'robot_description',
            '-x', LaunchConfiguration('spawn_x'),
            '-y', LaunchConfiguration('spawn_y'),
            '-z', LaunchConfiguration('spawn_z'),
            '-Y', LaunchConfiguration('spawn_yaw'),
        ],
        output='screen'
    )

    slam = TimerAction(
        period=5.0,
        actions=[Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_params, {'use_sim_time': True}],
        )]
    )

    nav2 = TimerAction(
        period=8.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_pkg, 'launch', 'navigation_launch.py')
            ),
            launch_arguments={
                'use_sim_time': 'true',
                'params_file':  nav2_params,
            }.items()
        )]
    )

    return LaunchDescription([
        world_arg, spawn_x, spawn_y, spawn_z, spawn_yaw,
        gazebo, rsp, spawn, slam, nav2,
    ])
