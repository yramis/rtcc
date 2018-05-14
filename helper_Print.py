import sys

bold        = '\033[1m'
underline   = '\033[4m'
red         = '\033[31m'
green       = '\033[92m'
yellow      = '\033[93m'
blue        = '\033[94m'
purple      = '\033[95m'
cyan        = '\033[96m'
end         = '\033[0m'
colors      = [red,green,yellow,blue,purple,cyan]

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

