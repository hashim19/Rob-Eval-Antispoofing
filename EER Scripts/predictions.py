def evaluate_classifications(file_path):
    total_count = 0
    incorrect_count = 0

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            label = parts[2]  
            score = float(parts[-1])

            total_count += 1
            if label == 'spoof' and score > 0:
                incorrect_count += 1

    accuracy = ((total_count - incorrect_count) / total_count) * 100
    print(f"{incorrect_count}/{total_count} classifications are incorrect.")
    print(f"Accuracy: {accuracy:.2f}%")


file_path = 'C:/Users/aesal/OneDrive/Documents/Rob-Antispoofing/Rob-Eval-Antispoofing/EER Scripts/predictions.txt'
evaluate_classifications(file_path)