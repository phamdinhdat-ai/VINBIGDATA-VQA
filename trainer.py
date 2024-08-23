import torch 
import os 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, f1_score
import pickle 


model = ''
train_loader = ''
val_loader = ''
batch_size = 10 
device= 'cuda:0'
vocab_swap = []
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW()



num_epochs = 30
print_every = 200


train_losses = []
val_losses = []
train_accuracy_scores = []
val_accuracy_scores = []
train_f1_scores = []
val_f1_scores = []
train_vqa_scores = []
val_vqa_scores = []

def vqa_accuracy_score(preds, answer_lists):
    acc_sum = 0
    for pred, answer_list in zip(preds, answer_lists):
        acc_sum += min(answer_list.count(pred) / 3, 1)
    return acc_sum / len(preds)

def trainer(model,
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            vocab_swap, 
            num_epochs = 10,
            batch_size = 256, 
            earlystop = True,
            log_results = True, 
            save_checkpoint = True
            ):
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_every = 200
    train_losses = []
    val_losses = []
    train_accuracy_scores = []
    val_accuracy_scores = []
    train_f1_scores = []
    val_f1_scores = []
    train_vqa_scores = []
    val_vqa_scores = []
    history = dict()
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        total_train_accuracy = 0.0
        total_train_f1 = 0.0
        total_train_vqa = 0.0
        total_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images, questions, answers_list, answer_types, answerables, answers = batch
            if len(images) == batch_size:
                predicted_tokens, ans_embedds = model(images.to(device), questions, answers, mask=True)
                predicted_tokens = predicted_tokens.float()
                ans_embedds = ans_embedds.long()

                # Predictions
                predictions = torch.argmax(predicted_tokens, dim=2)

                # Flattening the tensors for metric calculations
                predictions_flat = predictions.view(-1).cpu().numpy()
                answers_flat = ans_embedds.view(-1).cpu().numpy()
                answer_lists = [answer_list for answer_list in answers_list]

                # Calculate accuracy, F1 score, and VQA score
                accuracy = accuracy_score(answers_flat, predictions_flat)
                f1 = f1_score(answers_flat, predictions_flat, average='macro')

                preds = []
                for i in range(batch_size):
                    predicted_sentence = "".join([vocab_swap[idx.item()] for idx in predictions[i] if idx.item() != 50256])
                    predicted_sentence = predicted_sentence.replace('Ġ', ' ').strip()
                    preds.append(predicted_sentence)

                vqa_score = vqa_accuracy_score(preds, answer_lists)

                total_train_accuracy += accuracy
                total_train_f1 += f1
                total_train_vqa += vqa_score

                # Compute loss
                loss = criterion(predicted_tokens.permute(0, 2, 1), ans_embedds)
                valid_indicies = torch.where(ans_embedds == 1, False, True)
                loss = loss.sum() / valid_indicies.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                total_train_batches += 1

                if (batch_idx + 1) % print_every == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    print(f"Train Accuracy: {accuracy:.4f}")
                    print(f"Train F1 Score: {f1:.4f}")
                    print(f"Train VQA Score: {vqa_score:.4f}")

                    # Print example predictions
                    for i in range(batch_size):
                        predicted_sentence = "".join([vocab_swap[idx.item()] for idx in predictions[i] if idx.item() != 50256])
                        predicted_sentence = predicted_sentence.replace('Ġ', ' ').strip()
                        print(f"Question: {questions[i]}")
                        print(f"Answer: {answers[i]}")
                        print(f"Answer Prediction: {predicted_sentence}")
                    print("\n")

        avg_train_loss = total_train_loss / total_train_batches
        avg_train_accuracy = total_train_accuracy / total_train_batches
        avg_train_f1 = total_train_f1 / total_train_batches
        avg_train_vqa = total_train_vqa / total_train_batches

        train_losses.append(avg_train_loss)
        train_accuracy_scores.append(avg_train_accuracy)
        train_f1_scores.append(avg_train_f1)
        train_vqa_scores.append(avg_train_vqa)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        total_val_accuracy = 0.0
        total_val_f1 = 0.0
        total_val_vqa = 0.0
        total_val_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images, questions, answers_list, answer_types, answerables, answers = batch
                if len(images) == batch_size:
                    predicted_tokens, ans_embedds = model(images.to(device), questions, answers, mask=True)
                    predicted_tokens = predicted_tokens.float()
                    ans_embedds = ans_embedds.long()

                    # Predictions
                    predictions = torch.argmax(predicted_tokens, dim=2)

                    # Flattening the tensors for metric calculations
                    predictions_flat = predictions.view(-1).cpu().numpy()
                    answers_flat = ans_embedds.view(-1).cpu().numpy()
                    answer_lists = [answer_list for answer_list in answers_list]

                    # Calculate accuracy, F1 score, and VQA score
                    accuracy = accuracy_score(answers_flat, predictions_flat)
                    f1 = f1_score(answers_flat, predictions_flat, average='macro')

                    preds = []
                    for i in range(batch_size):
                        predicted_sentence = "".join([vocab_swap[idx.item()] for idx in predictions[i] if idx.item() != 50256])
                        predicted_sentence = predicted_sentence.replace('Ġ', ' ').strip()
                        preds.append(predicted_sentence)

                    vqa_score = vqa_accuracy_score(preds, answer_lists)

                    total_val_accuracy += accuracy
                    total_val_f1 += f1
                    total_val_vqa += vqa_score

                    # Compute loss
                    loss = criterion(predicted_tokens.permute(0, 2, 1), ans_embedds)
                    valid_indicies = torch.where(ans_embedds == 1, False, True)
                    loss = loss.sum() / valid_indicies.sum()

                    total_val_loss += loss.item()
                    total_val_batches += 1

        avg_val_loss = total_val_loss / total_val_batches
        avg_val_accuracy = total_val_accuracy / total_val_batches
        avg_val_f1 = total_val_f1 / total_val_batches
        avg_val_vqa = total_val_vqa / total_val_batches

        val_losses.append(avg_val_loss)
        val_accuracy_scores.append(avg_val_accuracy)
        val_f1_scores.append(avg_val_f1)
        val_vqa_scores.append(avg_val_vqa)
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Average Train Accuracy: {avg_train_accuracy:.4f}")
        print(f"Average Train F1 Score: {avg_train_f1:.4f}")
        print(f"Average Train VQA Score: {avg_train_vqa:.4f}")
        print(f"Average Val Loss: {avg_val_loss:.4f}")
        print(f"Average Val Accuracy: {avg_val_accuracy:.4f}")
        print(f"Average Val F1 Score: {avg_val_f1:.4f}")
        print(f"Average Val VQA Score: {avg_val_vqa:.4f}")
    history['loss'] = train_losses
    history['acc'] = train_accuracy_scores
    history['f1-score'] = train_f1_scores
    history['vqa-score'] = train_vqa_scores
    history['val_loss'] = val_losses
    history['val_acc'] = val_accuracy_scores
    history['val_f1_score'] = val_f1_scores
    history['val_vqa_score'] = val_vqa_scores
    return history



    