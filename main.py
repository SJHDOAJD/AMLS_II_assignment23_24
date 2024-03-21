import os
from A.ViT import preprocess_ViT
from A.ViT import model_ViT
from A.ViT import train_ViT
from A.ViT import evaluate_ViT
from A.ResNext import preprocess_ResNext
from A.ResNext import model_ResNext
from A.ResNext import train_ResNext
from A.ResNext import evaluate_ResNext
from A.ResNext1 import preprocess_ResNext1
from A.ResNext1 import model_ResNext1
from A.ResNext1 import train_ResNext1
from A.ResNext1 import evaluate_ResNext1


# ======================================================================================================================
# ViT method

train_dataset, valid_dataset, test_dataset = preprocess_ViT()
model1 = model_ViT()
acc_train, acc_valid, ls_train, ls_valid = train_ViT(model1, train_dataset, valid_dataset)
test_acc, test_loss = evaluate_ViT(model1, test_dataset)

print('ALL accuracy:{},{},{};ALL loss:{},{},{};'.format(acc_train, acc_valid, test_acc, ls_train, ls_valid, test_loss))

# ======================================================================================================================
# ResNext method (normal)

#train_dataset1, valid_dataset1, test_dataset1 = preprocess_ResNext()
#model3 = model_ResNext()
#acc_train2, acc_valid2, ls_train2, ls_valid2 = train_ResNext(model3, train_dataset1, valid_dataset1)
#test_acc2, test_loss2 = evaluate_ResNext(model3, test_dataset1)

#print('ALL accuracy:{},{},{};ALL loss:{},{},{};'.format(acc_train2, acc_valid2, test_acc2, ls_train2, ls_valid2, test_loss2))

# ======================================================================================================================
# ResNext method (with scheduler and weight)

#train_dataset2, valid_dataset2, test_dataset2 = preprocess_ResNext1()
#model4 = model_ResNext1()
#acc_train3, acc_valid3, ls_train3, ls_valid3 = train_ResNext1(model4, train_dataset2, valid_dataset2)
#test_acc3, test_loss3 = evaluate_ResNext1(model4, test_dataset2)

#print('ALL accuracy:{},{},{};ALL loss:{},{},{};'.format(acc_train3, acc_valid3, test_acc3, ls_train3, ls_valid3, test_loss3))

# ======================================================================================================================