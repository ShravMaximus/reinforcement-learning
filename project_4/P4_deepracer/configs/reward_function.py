#for head to bot and obstacle avoidance
import math

def reward_function(params):
    

    # base reward
    reward = 1e-3

    # params
    all_wheels_on_track= params.get('all_wheels_on_track', False)
    distance_from_center= params.get('distance_from_center', 0.0)
    track_width= params.get('track_width', 1.0)
    steering_angle = abs(params.get('steering_angle', 0.0))
    heading= params.get('heading', 0.0)
    progress= params.get('progress', 0.0)
    steps =params.get('steps', 1)

    #x-y posy
    agent_x = params.get('x', 0.0)
    agent_y = params.get('y', 0.0)
    is_left_of_center = params.get('is_left_of_center', False)

    #obs
    objects_location = params.get('objects_location', [])
    closest_objects = params.get('closest_objects', [])
    objects_left_of_center = params.get('objects_left_of_center', [])

    if all_wheels_on_track and (0.5 *track_width - distance_from_center) > 0.05:
        reward += 1.0
    else:
        return 1e-3 

    if steps > 0:
        reward += (progress / steps) * 0.5

    # obs avoid
    if closest_objects and objects_location and objects_left_of_center:
        _, obj_idx = closest_objects
        if obj_idx < len(objects_location):
            obj_x, obj_y = objects_location[obj_idx]
            dist = math.sqrt((agent_x - obj_x) ** 2 + (agent_y - obj_y) ** 2)
            is_obj_left = objects_left_of_center[obj_idx]

            if dist < 0.8:
                if is_obj_left and params["steering_angle"] < -15:
                    reward += 1.0  # turn right
                elif not is_obj_left and params["steering_angle"] > 15:
                    reward += 1.0  # turn left
                else:
                    reward *= 0.4 

            if dist < 0.3:
                reward = 1e-3 

    # re-centre
    if distance_from_center < 0.1:
        reward += 0.3

    waypoints = params.get('waypoints', [])
    closest_wp_idx = params.get('closest_waypoints', [0, 1])[1]
    if closest_wp_idx < len(waypoints):
        next_wp = waypoints[closest_wp_idx]
        track_angle = math.degrees(math.atan2(next_wp[1] - agent_y, next_wp[0] - agent_x))
        angle_diff = abs(track_angle - heading)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        if angle_diff < 10:
            reward += 0.2 

    return float(reward)
