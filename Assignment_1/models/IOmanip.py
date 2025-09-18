# Helper functions for better formatting
def print_section_header(title):
    print("\n" + "-"*80)
    print(f"  {title.upper()}")
    print("-"*80)

def add_spacing():
    print("\n" * 2)

def print_styled_box(content, title="", box_char="█", width=80):
    """Print content in a styled box"""
    print("\n")
    print(box_char * width)
    if title:
        print(f"{box_char} {title.center(width-4)} {box_char}")
        print(box_char * width)

    lines = content.strip().split('\n')
    for line in lines:
        padding = width - len(line) - 4
        print(f"{box_char} {line}{' ' * padding} {box_char}")
    print(box_char * width)
    print("\n")