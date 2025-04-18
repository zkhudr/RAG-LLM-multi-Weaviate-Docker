with open('index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

stack = []
for i, line in enumerate(lines, 1):
    if '<div' in line:
        stack.append(i)
    if '</div>' in line:
        if stack:
            stack.pop()
        else:
            print(f"Extra </div> found at line {i}")
