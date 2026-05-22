import time
import os

start = time.perf_counter()

BAUS_folder = "./3_alignement_force/"
Paty_folder = "./Paty_alignment/"
if len(os.listdir(Paty_folder)) != 0:
    for filename in sorted(os.listdir(BAUS_folder)):
        if filename.endswith("TextGrid"):
            print(filename)
            Paty_path = Paty_folder + filename[:-9] + ".wav/ctm.textgrid"

            txtgrid_path_BAUS = BAUS_folder + filename

            Paty_file = open(Paty_path, "r")
            Paty_data = Paty_file.read()
            Paty_data = Paty_data.replace('"B"', '"b"')
            Paty_data = Paty_data.replace('"D"', '"d"')
            Paty_data = Paty_data.replace('"F"', '"f"')
            Paty_data = Paty_data.replace('"G"', '"g"')
            Paty_data = Paty_data.replace('"J"', '"j"')
            Paty_data = Paty_data.replace('"K"', '"k"')
            Paty_data = Paty_data.replace('"L"', '"l"')
            Paty_data = Paty_data.replace('"M"', '"m"')
            Paty_data = Paty_data.replace('"N"', '"n"')
            Paty_data = Paty_data.replace('"P"', '"p"')
            Paty_data = Paty_data.replace('"S"', '"s"')
            Paty_data = Paty_data.replace('"T"', '"t"')
            Paty_data = Paty_data.replace('"V"', '"v"')
            Paty_data = Paty_data.replace('"W"', '"w"')
            Paty_data = Paty_data.replace('"Z"', '"z"')
            Paty_data = Paty_data.replace('"segmentation"', '"MAU"')
            Paty_data = Paty_data.replace('"<eps>"', '"<p:>"')
            Paty_data = Paty_data.replace('"AE"', '"@"')
            Paty_data = Paty_data.replace('"A~"', '"a~"')
            Paty_data = Paty_data.replace('"E~"', '"9~"')
            Paty_data = Paty_data.replace('"NJ"', '"nj"')
            Paty_data = Paty_data.replace('"SH"', '"S"')
            Paty_data = Paty_data.replace('"ZH"', '"Z"')

            Paty_data = Paty_data.partition("item [1]")[2]
            Paty_file.close()
            BAUS_file = open(txtgrid_path_BAUS, "r")
            BAUS_data = BAUS_file.read()
            BAUS_data = "".join(BAUS_data.partition("item [3]")[:2])
            new_data = BAUS_data + Paty_data
            BAUS_file.close()
            BAUS_file_write = open(txtgrid_path_BAUS, "w")
            BAUS_file_write.write(new_data)
            BAUS_file_write.close()

    print("Time elapsed during BAUS_to_Paty.py :", time.perf_counter() - start)


def to_paty(textgrid):
    conversion_table = {
        '"B"': '"b"',
        '"D"': '"d"',
        '"F"': '"f"',
        '"G"': '"g"',
        '"J"': '"j"',
        '"K"': '"k"',
        '"L"': '"l"',
        '"M"': '"m"',
        '"N"': '"n"',
        '"P"': '"p"',
        '"S"': '"s"',
        '"T"': '"t"',
        '"V"': '"v"',
        '"W"': '"w"',
        '"Z"': '"z"',
        '"segmentation"': '"MAU"',
        '"<eps>"': '"<p:>"',
        '"AE"': '"@"',
        '"A~"': '"a~"',
        '"E~"': '"9~"',
        '"NJ"': '"nj"',
        '"SH"': '"S"',
        '"ZH"': '"Z"'
    }

    for key, value in conversion_table.items():
        textgrid = textgrid.replace(key, value)
    textgrid = textgrid.partition("item [1]")[2]