"""Run planeslam in real-time with AirSim in Blocks environment

Settings: 'settings.json'

"""

import planeslam.airsim.setup_path
import airsim
import time

from planeslam.airsim.lidar import get_lidar_PC, reset

# Data collection parameters
interval = 1.0  # [s]
num_collections = 10 

# Trajectory parameters
z = 5.0  # flight altitude [m]
speed = 2.0  # flight speed [m/s]
lookahead = -1  # default = -1 (auto-decide)
adaptive_lookahead = 0  # default = 0 (auto-decide)
timeout = 10  # [s]

# MoveOnPath settings (https://microsoft.github.io/AirSim/apis/#apis-for-multirotor)
# drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom  # Don't care where front points
#                                                        as opposed to "airsim.DrivetrainType.ForwardOnly"
drivetrain = airsim.DrivetrainType.ForwardOnly
yaw_mode = airsim.YawMode(False, 0)  # (yaw_or_rate, is_rate)

# Load pre-planned trajectory
path = []
path.append([airsim.Vector3r(0, -60, -z),
             airsim.Vector3r(-50, -60, -z),
             airsim.Vector3r(-50, 0, -z),
             airsim.Vector3r(0, 0, -z)])


if __name__ == "__main__":

    # Connect to client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Arming drones...")
    client.enableApiControl(True)
    client.armDisarm(True)

    # Take off
    print("Taking off...")
    cmd = client.takeoffAsync()
    cmd.join()

    print("Ascending to hover altitude...")
    cmd = client.moveToZAsync(z=-z, velocity=1)
    cmd.join()

    # Begin flying
    try:
        print("Starting trajectory...")
        # Move command
        cmd = client.moveOnPathAsync(path, speed, timeout, drivetrain, 
            yaw_mode, lookahead, adaptive_lookahead)

        # Data collection
        for i in range(num_collections):
            P = get_lidar_PC(client)
            print('lidar points: ' + str(P.shape))
            time.sleep(interval)

        cmd.join()

    except KeyboardInterrupt:
        print("Interrupted")
        reset(client)

    # Done
    reset(client)
    print("Done")