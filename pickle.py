import pickle

# List of available modes
modes = ["GBB", "CSL", "KLD_", "KFIOU"]

# Default mode
mode = 'GBB'

print("Select a loss/IoU mode:")
for i, m in enumerate(modes):
    print(f"{i+1}. {m}")

try:
    user_input = input(f"Enter the number corresponding to your choice (default: {mode}): ")

    if user_input.strip(): # Check if the input is not empty
        selected_index = int(user_input) - 1 # Convert to 0-based index
        if 0 <= selected_index < len(modes):
            mode = modes[selected_index]
        else:
            print(f"Invalid number. Sticking with default mode: {mode}")
    else:
        print(f"No input provided. Sticking with default mode: {mode}")

except ValueError:
    print(f"Invalid input. Please enter a number. Sticking with default mode: {mode}")

print(f"{mode} loss/IoU mode selected")

# Assign the selected mode (as a string) to myvar
myvar = mode

with open("datacar.pkl", "wb") as f:
    pickle.dump(myvar, f)

print("Variable saved to datacar.pkl")
