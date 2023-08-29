import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import os
import json
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 设置训练的entities文件夹路径
entities_folder = 'supermarket_receipt/train'

# 设置验证的val文件夹路径
val_folder = 'supermarket_receipt/val'

# 获取模型所在目录
model_dir = os.path.dirname(os.path.realpath(__file__))
entities_path = os.path.join(model_dir, entities_folder)
#加入验证集
val_path = os.path.join(model_dir, val_folder)


#准备训练数据
train_receipts = []
for filename in os.listdir(entities_path):
    if filename.endswith('.json'):
        file_path = os.path.join(entities_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for doc in data["documents"]:
            text_list = [item["text"] for item in doc["document"] if item["label"] != "other"]
            label_list = [item["label"] for item in doc["document"] if item["label"] != "other" and item["label"] != "question"]
            label = " ".join(label_list)
            text = "\n".join(text_list)
            train_receipts.append({"text": text, "label": label})

#准备验证数据
validation_receipts = []
for filename in os.listdir(val_path):  
    if filename.endswith('.json'):
        file_path = os.path.join(val_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for doc in data["documents"]:
            text_list = [item["text"] for item in doc["document"] if item["label"] != "other"]
            label_list = [item["label"] for item in doc["document"] if item["label"] != "other" and item["label"] != "question"]
            label = " ".join(label_list)
            text = "\n".join(text_list)
            validation_receipts.append({"text": text, "label": label})

# 创建tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# 添加Pseudo Token到tokenizer
pseudo_tokens = ["[P0:COMPANY]", "[P1:ADDRESS]", "[P2:TIME]", "[P3:TOTAL]"]
tokenizer.add_tokens(pseudo_tokens)

# 找出最长的文本长度
max_length = max(len(tokenizer.tokenize(receipt['text'])) for receipt in train_receipts)
# 定义映射字典，将字符串标签映射到数值标签
label_mapping = {
    "header": ["[P0:COMPANY]", "[P1:ADDRESS]"],
    "answer": ["[P2:TIME]", "[P3:TOTAL]"],
}

modified_input_ids = []
modified_attention_masks = []
labels = []  

for receipt in train_receipts:
    text = receipt['text']  # 提取text值
    label = receipt['label']  # 提取label值

    # 根据映射字典逐步选择新的标签
    new_label = label
    for key, values in label_mapping.items():
        if key in new_label:
            random.shuffle(values)  # 随机排序映射值
            random_value = values[0]  # 选择第一个值
            new_label = new_label.replace(key, random_value)
    
    # 如果new_label为空，跳过该样本
    if not new_label:
        continue

    # 在这里再进行其他标签处理，例如将 "[P0:COMPANY]" 替换为 "COMPANY"
    new_label = new_label.replace("[P0:COMPANY]", "COMPANY")
    new_label = new_label.replace("[P1:ADDRESS]", "ADDRESS")
    new_label = new_label.replace("[P2:TIME]", "TIME")
    new_label = new_label.replace("[P3:TOTAL]", "TOTAL")
    # 将新的标签添加到列表中
    labels.append(new_label)
    # 将text和label组合成一个字符串
    combined_text = f"{text}\n{labels}"

    # 使用tokenizer对组合后的文本进行编码
    encoded = tokenizer(combined_text, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
    modified_input_ids.append(encoded["input_ids"].squeeze()) 
    modified_attention_masks.append(encoded["attention_mask"].squeeze())

# 将列表中的每个元素转换为张量
input_ids_tensor = torch.stack(modified_input_ids)
attention_masks_tensor = torch.stack(modified_attention_masks)
# 将标签转换为数值标签
label_to_id = {"COMPANY": 0, "ADDRESS": 1, "TIME": 2, "TOTAL": 3}

# 检查labels列表中是否有新的标签需要添加到label_to_id字典中
for label in labels:
    if label not in label_to_id:
        new_id = len(label_to_id)  
        label_to_id[label] = new_id
# 将标签转换为数值标签
labels_to_ids = {label: idx for idx, label in enumerate(label_to_id)}
labels_tensor = torch.tensor([labels_to_ids[label] for label in labels], dtype=torch.long)

#预处理和分词的验证集数据
validation_input_ids = []
validation_attention_masks = []
validation_labels = []  

for receipt in validation_receipts:
    text = receipt['text']  # 提取text值
    label = receipt['label']  # 提取label值

    # 根据映射字典逐步选择新的标签
    new_label = label
    for key, values in label_mapping.items():
        if key in new_label:
            random.shuffle(values)  # 随机排序映射值
            random_value = values[0]  # 选择第一个值
            new_label = new_label.replace(key, random_value)
    
    # 如果new_label为空，跳过该样本
    if not new_label:
        continue

    # 在这里再进行其他标签处理,将"[P0:COMPANY]" 替换为 "COMPANY"
    new_label = new_label.replace("[P0:COMPANY]", "COMPANY")
    new_label = new_label.replace("[P1:ADDRESS]", "ADDRESS")
    new_label = new_label.replace("[P2:TIME]", "TIME")
    new_label = new_label.replace("[P3:TOTAL]", "TOTAL")
    # 将新的标签添加到列表中
    validation_labels.append(new_label)
    # 将text和label组合成一个字符串
    combined_text = f"{text}\n{validation_labels}"

    # 使用tokenizer对组合后的文本进行编码
    encoded = tokenizer(combined_text, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
    validation_input_ids.append(encoded["input_ids"].squeeze())  
    validation_attention_masks.append(encoded["attention_mask"].squeeze())

# 将列表中的每个元素转换为张量
validation_input_ids_tensor = torch.stack(validation_input_ids)
validation_attention_masks_tensor = torch.stack(validation_attention_masks)
# 将标签转换为数值标签
label_to_id = {"COMPANY": 0, "ADDRESS": 1, "TIME": 2, "TOTAL": 3}

# 检查labels列表中是否有新的标签需要添加到label_to_id字典中
for label in validation_labels:
    if label not in label_to_id:
        new_id = len(label_to_id) 
        label_to_id[label] = new_id
# 将标签转换为数值标签
labels_to_ids = {label: idx for idx, label in enumerate(label_to_id)}
validation_labels_tensor = torch.tensor([labels_to_ids[label] for label in validation_labels], dtype=torch.long)


# 创建模型
model = BertForMaskedLM.from_pretrained(model_name)
#将Pseudo Token映射为可训练的embedding tensors
pseudo_token_ids = tokenizer.convert_tokens_to_ids(pseudo_tokens)
for token_id in pseudo_token_ids:
    model.bert.embeddings.word_embeddings.weight.data[token_id] = torch.randn(model.config.hidden_size)

#添加Dropout层
model.bert.embeddings.dropout = torch.nn.Dropout(0.7)

#批大小
batch_size =32

# 创建TensorDataset
train_dataset = TensorDataset(input_ids_tensor, attention_masks_tensor, labels_tensor)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#创建验证集的DateLoader
validation_dataset = TensorDataset(validation_input_ids_tensor, validation_attention_masks_tensor, validation_labels_tensor)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)  

# 训练循环
model.to('cuda')
num_epochs = 500
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-5)
criterion = torch.nn.CrossEntropyLoss()
  
best_accuracy = 0.0
best_epoch = 0
#记录每个周期的损失和准确率
losses = []
accuracies = []
validation_losses = []
validation_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        batch_input_ids, batch_attention_masks, batch_labels = batch
        optimizer.zero_grad()

        # 将数据移动到GPU上
        batch_input_ids = batch_input_ids.to('cuda')
        batch_attention_masks = batch_attention_masks.to('cuda')
        batch_labels = batch_labels.to('cuda')
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
        logits = outputs.logits

        # 在 logits 上应用 softmax
        probs = F.softmax(logits, dim=-1)

        # 计算交叉熵损失
        expanded_labels = batch_labels.view(-1, 1).repeat(1, max_length)  
        flat_logits = logits.view(-1, logits.shape[-1])
        flat_labels = expanded_labels.view(-1)
        loss = criterion(flat_logits, flat_labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == expanded_labels).float().mean()
        total_accuracy += accuracy.item()
    
    average_loss = total_loss / num_batches
    average_accuracy = total_accuracy / num_batches

    #评估验证集
    model.eval()  
    validation_loss = 0.0
    validation_accuracy = 0.0
    with torch.no_grad():
        for batch in validation_dataloader:
            batch_input_ids, batch_attention_masks, batch_labels = batch
            batch_input_ids = batch_input_ids.to('cuda')
            batch_attention_masks = batch_attention_masks.to('cuda')
            batch_labels = batch_labels.to('cuda')
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
            logits = outputs.logits

            expanded_labels = batch_labels.view(-1, 1).repeat(1, max_length)
            flat_logits = logits.view(-1, logits.shape[-1])
            flat_labels = expanded_labels.view(-1)
            loss = criterion(flat_logits, flat_labels)
            validation_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == expanded_labels).float().mean()
            validation_accuracy += accuracy.item()

    average_validation_loss = validation_loss / len(validation_dataloader)
    average_validation_accuracy = validation_accuracy / len(validation_dataloader)

    print(f"Epoch {epoch + 1}, Train Loss: {average_loss:.4f}, Train Accuracy: {average_accuracy:.4f}, "
          f"Validation Loss: {average_validation_loss:.4f}, Validation Accuracy: {average_validation_accuracy:.4f}")
    # 添加损失和准确率到列表中
    losses.append(average_loss)
    accuracies.append(average_accuracy)
    validation_losses.append(average_validation_loss)  # 将验证损失添加到列表
    validation_accuracies.append(average_validation_accuracy)  # 将验证准确率添加到列表

# 将损失和准确率数据保存为CSV文件
csv_filename = "loss_and_accuracy.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Epoch", "Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"])  
    for epoch, loss, accuracy, val_loss, val_accuracy in zip(range(1, num_epochs + 1), losses, accuracies, validation_losses, validation_accuracies):
        csvwriter.writerow([epoch, loss, accuracy, val_loss, val_accuracy])
# # 绘制损失曲线
# plt.figure(figsize=(10, 5))
# plt.plot(losses, label='Loss', color='blue')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# #绘制准确率曲线
# plt.figure(figsize=(10, 5))
# plt.plot(accuracies, label='Accuracy', color='green')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()

# 保存训练好的模型
output_dir = "ptuning_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)