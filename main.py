import openai
import pandas as pd
import os
import json
import matplotlib.pyplot as plt


##training file "id": "file-LrwILn6VBeUCD7LzmIWhUlMY"
##validation file  "id": "file-nUpN2JIBxVBnRY5ZFoiPAcCM"
#FT job "id": "ft-r2VhoBHH3s0F9AGfQStPWbwn"
#model: davinci:ft-personal:linkedinlearningv1-2023-03-08-01-40-01


openai.api_key = os.environ.get("OPENAI_API_KEY")

#fine-tune model
# ftmodel = openai.FineTune.create(
#     training_file = "file-LrwILn6VBeUCD7LzmIWhUlMY",
#     validation_file = "file-nUpN2JIBxVBnRY5ZFoiPAcCM",
#     model = "davinci",
#     classification_n_classes = 2,
#     classification_positive_class = "yes",
#     suffix = "linkedinlearningv1",
#     compute_classification_metrics = True
# )


#query model with test data and save the results
#relevant output of saved variable is result["choices"][0]["text"]
testdata = pd.read_json("test.jsonl", lines = True)
testresult = pd.DataFrame(columns = ["actual", "predicted", "difference"])
results = []


#####
# for i,row in testdata.iterrows():
#     prompt = row["prompt"] #set prompt

#     result = openai.Completion.create(
#         model="davinci:ft-personal:linkedinlearningv1-2023-03-08-01-40-01",
#         prompt=prompt,
#         max_tokens=1,
#         temperature=0
#     )
#     #pull actual, result, and difference. results is a Plan B
#     actual = row["completion"]
#     result = result["choices"][0]["text"]
#     diff = ""
#     diff = 0 if actual == result else 1

#     #get row setup
#     result_row = {
#         "actual": actual,
#         "predicted": result,
#         "difference": diff
#     }
#     #append to bottom of testresult
#     testresult.loc[len(testresult)] = result_row


# testresult.to_csv("testresults.csv")
# ####

answers = pd.read_csv('testresults.csv')

correct = 0
false_pos = 0
false_neg = 0

for i,row in answers.iterrows():
    if row["difference"] == 0:
        correct +=1
    elif row["difference"] == 1:
        if row["actual"] == "yes":
            false_neg +=1
        else:
            false_pos +=1


print("correct:")
print(correct)
print("false positive:")
print(false_pos)
print("false negative:")
print(false_neg)


# Create a list of labels for the x-axis
labels = ['Correct', 'False Positive', 'False Negative']

# Create a list of values for the y-axis
values = [correct, false_pos, false_neg]

# Create the bar graph
plt.bar(labels, values)

# Add a title to the graph
plt.title('LinkedIn Learning License Prediction Summary')

# Add labels to the x and y axes
plt.xlabel('Category')
plt.ylabel('Count')

# Display the graph
plt.show()




