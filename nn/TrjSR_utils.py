# code ref: TrjSR

from PIL import Image, ImageDraw
from collections import Counter

# For LR image: transform the trajectory point into cell id, return the sequence of cell id
def traj2cell_test_lr(seq, lon_range, lat_range, imgsize_x_lr, imgsize_y_lr, pixelrange_lr):
    cell_seq = []
    for j in range(len(seq)): # for each trajectory point
        # x, y = lonlat2meters(seq[j][0], seq[j][1])  # transform the coordinates into meters
        x, y = seq[j][0], seq[j][1] # yc: sequence is in merc space, no need conversion
        cell_seq.append(coord2cell_lr(x, y, lon_range, lat_range, imgsize_x_lr, imgsize_y_lr, pixelrange_lr)) # the cell id for this trajectory point
    return cell_seq


def coord2cell_lr(x, y, lon_range, lat_range, imgsize_x_lr, imgsize_y_lr, pixelrange_lr):
    xoffset = (x - lon_range[0]) / (lon_range[1] - lon_range[0]) * imgsize_x_lr / pixelrange_lr
    yoffset = (lat_range[1] - y) / (lat_range[1] - lat_range[0]) * imgsize_y_lr / pixelrange_lr
    xoffset = int(xoffset)
    yoffset = int(yoffset)
    tmp = imgsize_x_lr / pixelrange_lr
    return yoffset * tmp + xoffset


# find the anchor points of the cell
def cell2anchor(xoffset, yoffset, pixel):
    left_upper_point_x = xoffset * pixel
    left_upper_point_y = yoffset * pixel
    right_lower_point_x = left_upper_point_x + pixel - 1
    right_lower_point_y = left_upper_point_y + pixel - 1
    return (left_upper_point_x, left_upper_point_y), (right_lower_point_x, right_lower_point_y)


# map cell id into pixel value of low-resolution image
def draw_lr(seq, imgsize_x_lr, imgsize_y_lr, pixelrange_lr):
    img = Image.new("L", (imgsize_x_lr, imgsize_y_lr))
    cellset = Counter(seq).keys() # all the different cell
    occurrence = Counter(seq).values() # the number of the occurrences of each cell
    for i, cell in enumerate(cellset):
        xoffset = cell % (imgsize_x_lr/pixelrange_lr)
        yoffset = cell // (imgsize_x_lr/pixelrange_lr)
        left_upper_point, right_lower_point = cell2anchor(xoffset, yoffset, pixelrange_lr)
        grayscale = 105 + list(occurrence)[i] * 50 if list(occurrence)[i] < 4 else 255
        shape = [left_upper_point, right_lower_point]
        ImageDraw.Draw(img).rectangle(shape, fill=(grayscale))
    # img.save("./scp/lr/{}.png".format(len(seq)))
    return img
