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

if sys.stdout.isatty():     # Wrapper to ensure the Printout is only colorized in real terminal
    def Print(obj):
        print(obj)
        return
else:
    def Print(obj):
        # Check for color applied to whole line
        colorized = False
        for color in colors:
            if color in obj:
                colorized = True
        if colorized:
            string = obj[5:-4]
            # Check if second color applied
            for color2 in colors:
                if color2 in string:
                    string = string.replace(color2,'')
            print(string,flush=True)
        else:
            print(obj,flush=True)
        return

