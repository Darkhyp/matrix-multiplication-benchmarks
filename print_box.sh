#!/usr/bin/env bash

print_box() {
    local text="$1"
    local width="${2:-80}"
    local text_width=${#text}

    # Include the two spaces surrounding the text
    (( text_width += 1 ))

    # Ensure the box is wide enough
    (( text_width > width )) && width=$text_width

    local line
    printf -v line '%*s' "$width" ''
    line=${line// /─}

    printf '┌%s┐\n' "$line"
    printf '│ %s' "$text"
    printf '%*s│\n' "$((width - text_width))" ''
    printf '└%s┘\n' "$line"
}
