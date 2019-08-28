
pandoc \
    lee_workshop_notes.rst \
    --variable "papersize=a4" \
    --variable "fontfamily=times" \
    --variable "geometry=margin=1in" \
    -o lee_workshop_notes.pdf
