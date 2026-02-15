import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('jasmitte_proj4')
    
    # 1. Localization Node
    loc_node = Node(
        package='jasmitte_proj4',
        executable='localization_node',
        name='localization_node',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # 2. RViz Node
    rviz_config_path = os.path.join(pkg_share, 'rviz', 'proj4.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        loc_node,
        rviz_node
    ])
