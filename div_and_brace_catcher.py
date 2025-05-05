import argparse

def div_and_brace_catcher(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return

    div_stack = []  # Stack to track <div> tag positions
    brace_stack = []  # Stack to track { brace positions

    for i, line in enumerate(lines, 1):
        # Check for opening div tags
        if '<div' in line:
            div_stack.append(i)
        
        # Check for closing div tags
        if '</div>' in line:
            if div_stack:
                div_stack.pop()  # Matched div, pop the opening from stack
            else:
                print(f"Extra </div> found at line {i}")
        
        # Check for opening curly braces
        if '{' in line:
            brace_stack.append(i)
        
        # Check for closing curly braces
        if '}' in line:
            if brace_stack:
                brace_stack.pop()  # Matched brace, pop the opening from stack
            else:
                print(f"Extra }} found at line {i}")
    
    # Check for unmatched opening div tags
    if div_stack:
        for line_number in div_stack:
            print(f"Unmatched <div> at line {line_number}")
    
    # Check for unmatched opening curly braces
    if brace_stack:
        for line_number in brace_stack:
            print(f"Unmatched {{ at line {line_number}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Check for unbalanced div tags and braces in a file.")
    parser.add_argument('filename', type=str, help="The filename to check for unbalanced div tags and braces")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call div_and_brace_catcher with the provided filename
    div_and_brace_catcher(args.filename)

if __name__ == '__main__':
    main()
