import glob
import re
from collections import defaultdict

for dir in ['mlx', 'num']:
    # Find all Python files in the directory
    files = sorted(glob.glob(f"{dir}/*.py"))

    # Store dependencies: which file depends on which
    dependencies = defaultdict(set)

    # Regex pattern to match import statements (both `import X` and `from X import Y`)
    import_pattern = re.compile(r'^\s*(?:from|import)\s+([\w\.]+)')

    # Extract dependencies from each file
    file_modules = {f[:-3]: f for f in files}  # Map module name to filename

    for file in files:
        module_name = file[:-3]  # Remove `.py`
        with open(file, "r") as f:
            for line in f:
                match = import_pattern.match(line)
                if match:
                    imported_module = match.group(1).split('.')[0]  # Get base module
                    if imported_module in file_modules and imported_module != module_name:
                        dependencies[module_name].add(imported_module)

    # Topological sorting (Kahn's algorithm)
    sorted_files = []
    indegree = {f[:-3]: 0 for f in files}  # Track dependencies

    # Count incoming edges (dependencies)
    for deps in dependencies.values():
        for dep in deps:
            indegree[dep] += 1

    # Start with independent files (no dependencies)
    queue = [f for f in indegree if indegree[f] == 0]

    while queue:
        module = queue.pop(0)
        sorted_files.append(file_modules[module])
        
        for dependent in dependencies:
            if module in dependencies[dependent]:
                dependencies[dependent].remove(module)
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)

    # Ensure all files are included (fallback in case of cyclic dependencies)
    remaining_files = [file_modules[f] for f in indegree if file_modules[f] not in sorted_files]
    final_order = sorted_files + remaining_files

    # Merge files in determined order
    output_file = f"{dir}/merged_output.py"
    with open(output_file, "w") as outfile:
        for filename in final_order:
            with open(filename, "r") as infile:
                content = infile.readlines()
                # Remove local imports
                filtered_content = []
                for line in content:
                    words = line.strip().split()  # Tokenize the line
                    if words and (words[0] == "import" or words[0] == "from"):
                        module_name = words[1].split(".")[0]  # Get base module (ignore submodules)
                        if module_name in file_modules:
                            continue  # Skip this line (remove it)
                    
                    filtered_content.append(line)  # Keep non-local imports

                
                outfile.write(f"# --- Start of {filename} ---\n\n")
                outfile.writelines(content)  # Write remaining content (no imports)
                outfile.write(f"\n# --- End of {filename} ---\n\n")

    print(f"Python files merged into {output_file} in dependency order (imports removed).")