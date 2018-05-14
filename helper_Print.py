import sys

if sys.stdout.isatty():
    def Print(obj):  # Wrapper to ensure the Printout is only colorized in real terminal
        print(obj)
        return
else:
    def Print(obj):
        colorized = False
        for color in colors:
            if color in obj:
                colorized = True
        if colorized:
            string = obj[5:-4]
            print(string)
        else:
            print(obj)
        return

