## testing if load model works

from simpletransformers.classification import ClassificationModel

text = 'People won’t admit they’re going to vote for him. I don’t want the person that’s behind your public Facebook account, I want the person that’s behind your troll account. When I look at Pennsylvania, for example, I’ve got Biden up by one point, but I don’t think Biden is going to win Pennsylvania. I think Trump is probably going to win it. I think Trump will out-perform our polls by a point or two.'

model = ClassificationModel('bert', 'model', use_cuda=False)
prediction = model.predict(text)
print(prediction)