#!/usr/bin/env python3
"""
Interactive CLI for the robot.
Type any natural language command, robot executes, prompts for next.

Usage:
  ros2 run semantic_mobility_nav robot_cli
"""

import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RobotCLI(Node):
    def __init__(self):
        super().__init__('robot_cli')
        self.cmd_pub = self.create_publisher(String, '/task/command', 10)
        self.create_subscription(String, '/task/status', self._status_cb, 10)

        self._done = threading.Event()
        self._last_status = ''

    def _status_cb(self, msg):
        status = msg.data
        self._last_status = status

        if status.startswith('step:'):
            print(f'  -> {status[5:]}')
        elif status.startswith('planning:'):
            print(f'  [planning] Parsing command...')
        elif status.startswith('searching:'):
            print(f'  [searching] Looking for: {status[10:]}')
        elif status.startswith('found:'):
            print(f'  [found] Detected: {status[6:]} — approaching...')
        elif status.startswith('approaching:'):
            print(f'  [approaching] {status[12:]}')
        elif status.startswith('reached:'):
            print(f'  [REACHED] Successfully reached: {status[8:]}')
        elif status.startswith('report:'):
            print(f'  [report] {status[7:]}')
        elif status == 'done':
            if not self._done.is_set():   # suppress duplicate if failed already set it
                print('  [done] Task complete.')
            self._done.set()
        elif status.startswith('error:'):
            print(f'  [error] {status[6:]}')
            self._done.set()
        elif status.startswith('failed:'):
            print(f'  [failed] {status[7:]}')
            self._done.set()

    def send(self, command: str):
        self._done.clear()
        msg = String()
        msg.data = command
        self.cmd_pub.publish(msg)

    def wait_until_done(self, timeout=120.0):
        return self._done.wait(timeout=timeout)


def main(args=None):
    rclpy.init(args=args)
    node = RobotCLI()

    # Spin ROS in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print('\n🤖  Robot CLI ready. Type a command or "quit" to exit.\n')

    STOP_WORDS = {'stop', 'halt', 'cancel', 'abort', 'freeze', 'emergency stop'}

    try:
        while True:
            try:
                command = input('>> ').strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not command:
                continue
            if command.lower() in ('quit', 'exit', 'q'):
                break

            node.send(command)

            # Stop commands are fire-and-forget — don't block waiting
            if command.lower() in STOP_WORDS:
                print('  ⛔ Stopped.')
                continue

            done = node.wait_until_done(timeout=300.0)
            if not done:
                print('  ⚠ Timed out waiting for execution.')
            time.sleep(0.15)   # let callback finish printing before prompt reappears

    finally:
        node.destroy_node()
        rclpy.shutdown()
        print('\nBye.')


if __name__ == '__main__':
    main()
