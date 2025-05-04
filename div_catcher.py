with open('ragapp.js', 'r', encoding='utf-8') as f:
    lines = f.readlines()

stack = []
for i, line in enumerate(lines, 1):
    if '{' in line:
        stack.append(i)
    if '}' in line:
        if stack:
            stack.pop()
        else:
            print(f"Extra </div> found at line {i}")
