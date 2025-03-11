#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 15:27:19 2025

@author: mirjam
"""
import os
import inspect
import importlib.util
import sys

SRC_DIR = "src"  # Make sure this is the correct directory path

def load_module(file_path, module_name):
    """Dynamically loads a Python module from a file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def list_classes_and_methods():
    """Iterates over Python files in 'src/', extracts unique classes and methods."""
    print("## Class Overview\n")
    
    processed_classes = set()  # Track unique class names
    all_imports = set()  # Track imported module names

    # First, loop through the src directory and collect imports for each file
    for filename in os.listdir(SRC_DIR):
        if filename.endswith(".py"):
            file_path = os.path.join(SRC_DIR, filename)
            module_name = f"src.{filename[:-3]}"  # Use full package name

            try:
                module = load_module(file_path, module_name)

                # Check for imports in the module
                with open(file_path, 'r') as f:
                    file_content = f.read()
                    # Extract import statements (very basic version)
                    for line in file_content.splitlines():
                        if line.startswith('import ') or line.startswith('from '):
                            import_name = line.split()[1].split('.')[0]  # Extract the main module name
                            all_imports.add(import_name)

            except (ModuleNotFoundError, FileNotFoundError) as e:
                # Log errors to a file instead of printing them
                with open("error_log.txt", "a") as log_file:
                    log_file.write(f"Error loading {filename}: {str(e)}\n")
            except Exception as e:
                # Log other unexpected errors
                with open("error_log.txt", "a") as log_file:
                    log_file.write(f"Unexpected error with {filename}: {str(e)}\n")

    # Now loop through the src directory again, this time extracting classes
    for filename in os.listdir(SRC_DIR):
        if filename.endswith(".py"):
            file_path = os.path.join(SRC_DIR, filename)
            module_name = f"src.{filename[:-3]}"  # Use full package name

            try:
                # Only process files within the 'src' folder
                if not file_path.startswith(SRC_DIR):
                    continue

                module = load_module(file_path, module_name)

                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name not in processed_classes:
                        # Skip classes that are from imported modules
                        if obj.__module__.split('.')[0] in all_imports:
                            continue  # Skip classes from imports

                        # Skip external library classes
                        if obj.__module__.startswith(('sklearn', 'scipy', 'matplotlib')):
                            continue

                        processed_classes.add(name)  # Add to set to prevent duplicates
                        print(f"<details>\n  <summary>{name}</summary>\n")
                        for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                            print(f"- `{method_name}()`")
                        print("\n</details>\n")  # Close the collapsible section

            except (ModuleNotFoundError, FileNotFoundError) as e:
                # Log errors to a file instead of printing them
                with open("error_log.txt", "a") as log_file:
                    log_file.write(f"Error loading {filename}: {str(e)}\n")
            except Exception as e:
                # Log other unexpected errors
                with open("error_log.txt", "a") as log_file:
                    log_file.write(f"Unexpected error with {filename}: {str(e)}\n")


def list_functions_in_utils():
    """Iterates over utils*.py files in 'src/', extracts unique functions."""
    print("## Utility Functions Overview\n")
    
    processed_functions = set()  # Track unique function names

    for filename in os.listdir(SRC_DIR):
        if filename.startswith("utils") and filename.endswith(".py"):
            file_path = os.path.join(SRC_DIR, filename)
            module_name = f"src.{filename[:-3]}"  # Use full package name

            try:
                # Only process files within the 'src' folder
                if not file_path.startswith(SRC_DIR):
                    continue

                module = load_module(file_path, module_name)

                functions = [
                    name for name, obj in inspect.getmembers(module, inspect.isfunction)
                    if name not in processed_functions
                ]

                if functions:
                    print(f"<details>\n  <summary>{filename}</summary>\n")
                    for func in functions:
                        processed_functions.add(func)  # Mark function as processed
                        print(f"- `{func}()`")
                    print("\n</details>\n")  # Close the collapsible section

            except (ModuleNotFoundError, FileNotFoundError) as e:
                # Log errors to a file instead of printing them
                with open("error_log.txt", "a") as log_file:
                    log_file.write(f"Error loading {filename}: {str(e)}\n")
            except Exception as e:
                # Log other unexpected errors
                with open("error_log.txt", "a") as log_file:
                    log_file.write(f"Unexpected error with {filename}: {str(e)}\n")


list_classes_and_methods()                
list_functions_in_utils()
