import numpy as np


def distance_risk_score(distances, max_allowed_distance, min_allowed_distance):
    # Find the maximum and minimum distances
    max_distance = max(distances)
    min_distance = min(distances)

    # Initialize the distance risk score
    distance_risk_score = 0

    # Loop through each distance in the list
    for distance in distances:

        # Compute the distance risk score for this distance
        if distance > min_allowed_distance:
            # If the distance is greater than the minimum distance, add the normalized distance to the risk score
            allowed_distance = max_allowed_distance - min_allowed_distance
            distance_risk_score += (max_allowed_distance - distance) / (allowed_distance * len(distances))
        else:
            # If the distance is less than or equal to the minimum distance, subtract the normalized distance from the risk score
            distance_risk_score += (min_allowed_distance - distance) / (min_distance * len(distances))
    if min_distance> min_allowed_distance:
        distance_risk_score = distance_risk_score
    else:
        distance_risk_score = 1 + distance_risk_score

    return (distance_risk_score)