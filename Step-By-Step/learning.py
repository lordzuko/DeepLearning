# weight = 0.5
# input = 0.5
# goal_prediction = 0.8
#
# step_amount = 0.001
# for iteration in range(1101):
#
#     prediction = weight*input
#     error = (goal_prediction - prediction) ** 2
#     print("Error: " + str(error) + " Prediction: " + str(prediction))
#     up_prediction = input*(weight + step_amount)
#     up_error = (up_prediction - goal_prediction) ** 2
#
#     down_prediction = input*(weight - step_amount)
#     down_error = (down_prediction - goal_prediction) ** 2
#
#     if (down_error < up_error):
#         weight = weight - step_amount
#     else:
#         weight = weight + step_amount

weight = 0.5
goal_pred = 0.8
input = 0.5

for _ in range(20):
    pred = input * weight
    error = (pred - goal_pred) ** 2
    direction_and_amount = (pred - goal_pred) * input
    knob_weight = weight - direction_and_amount

    print("Error:" + str(error) + " Prediction:" + str(pred))