###############################################################################
#
# File              : Safety_Polygon_v1.py
# Author            : Jorge Sanchez & Eric Gonzalez
# Creation          : May-31-2024
# Description       : Creates flyrock polygons based on two radius using existing
#                     points on an object, for every point a 2D or 3D projection
#                     is created and then we get a polygon using convex hull.
###############################################################################
#          Maptek Pty Ltd (C) 2024 All rights reserved
###############################################################################


from maptek import vulcan
from maptek import vulcan_gui
from maptek.vulcan_gui import run_menu
import time
from math import sin, cos, tau
import numpy as np
from os import remove
import subprocess
from pathlib import Path
import os
import sys


def create_points(topo, radius, points, mode):
    r"""
    Project points using a 2D or 3D distance.

    Parameters
    ----------
    topo (str):     Path to triangulation.
    radius (float): Projection radius.
    points (list):  List of values [x,y,z]
    mode (int):     1 or 0  projection 2D/3D

    Returns
    ----------
    (list): List of vulcan points
    """

    point_list = []
    hull_points = []

    circle = create_circle(radius, 360)

    tri = None
    if mode == 0:
        tri = vulcan.triangulation(topo, "w")

    for p in points:
        x0 = p[0]
        y0 = p[1]
        z0 = p[2]

        # Translate the circle
        x = circle[0] + x0
        y = circle[1] + y0
        ref = vulcan.point()
        ref.x = x0
        ref.y = y0
        ref.z = z0

        # Find elevation in the topo
        for i in range(len(x)):
            temp_point = vulcan.point()
            temp_point.x = x[i]
            temp_point.y = y[i]
            temp_point.z = z0

            # Find 3D distance
            if mode == 0:
                temp_point.z = tri.get_elevation(x[i], y[i])
                temp_point = distance(tri, ref, temp_point, radius)

            # Save results
            hull_points.append((temp_point.x, temp_point.y, temp_point.z))

    # Create Convex Hull
    point_list = convex_hull(hull_points)
    return point_list


def distance(tri, p1, p2, radius):
    r"""
    Finds the endpoint that meets the specified 3D distance with recursion and
    the topographic triangulation.

    Parameters
    ----------
    tri (tri): Vulcan triangulation.
    p1 (point): Projection radius.
    p2 (point):  List of values [x,y,z]
    radius(float): 1 or 0  projection 2D/3D

    Returns
    ----------
    (point): Vulcan point
    """
    x1 = p1.x
    y1 = p1.y
    z1 = p1.z

    x2 = p2.x
    y2 = p2.y
    z2 = p2.z

    dist = ((x1 - x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5

    dx = (x2 - x1)
    dy = (y2 - y1)

    sf = radius / dist  # Scaling factor

    new_x = x1 + dx * sf
    new_y = y1 + dy * sf
    real_z = tri.get_elevation(new_x, new_y)

    # Flatten if necessary
    if real_z < z1:
        dist_2d = ((dx)**2 + (dy)**2)**0.5
        sf = radius / dist_2d
        new_x = x1 + dx * sf
        new_y = y1 + dy * sf

        end_point = vulcan.point()
        end_point.x = new_x
        end_point.y = new_y
        end_point.z = z1

        return end_point

    # Recursive distance correction
    new_dist = ((x1 - new_x)**2 + (y1-new_y)**2 + (z1-real_z)**2)**0.5
    if (new_dist - radius) > 5:
        new_point = vulcan.point()
        new_point.x = new_x
        new_point.y = new_y
        new_point.z = real_z

        distance(tri, p1, new_point, radius)

    end_point = vulcan.point()
    end_point.x = new_x
    end_point.y = new_y
    end_point.z = real_z

    return end_point


def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.

    https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain#Python
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 

    result = lower[:-1] + upper[:-1]

    point_list = []
    for p in result:
        point = vulcan.point()
        point.x = p[0]
        point.y = p[1]
        point.z = p[2]

        point_list.append(point)

    return point_list


def create_circle(radius, points):
    r"""
    Recursively finds the endpoint that meets the specified 3D distance using
    the topography triangulation.

    Parameters
    ----------
    radius (tri):
    points (int): Number of points in the circle.

    Returns
    ----------
    (numpy.Array): Array with X Y values for each point.
    """
    x0 = 0
    y0 = 0
    x = []
    y = []

    angle = 0
    angleStep = tau/float(points)
    for i in range(points):
        x.append(x0 + radius * cos(angle))
        y.append(y0 + radius * sin(angle))
        angle += angleStep

    circ = np.array([x, y])
    return circ


def selection():
    r"""
    Recursively finds the endpoint that meets the specified 3D distance using
    the topography triangulation.

    Parameters
    ----------
    None:


    Returns
    ----------
    (list): List of [X, Y, Z] values.
    """

    sel = vulcan_gui.selection("Select blast polygon")
    sel.criteria = (['LINE', 'POLYGON'])
    points = []

    for obj in sel:
        for p in obj:
            x = p.x
            y = p.y
            z = p.z
            points.append([x, y, z])

    if len(points) < 1:
        exit("No objects where selected")

    return points


def gpan():
    gpan_string = '''
    panel Main
    {
        title = " Safety Polygon Tool";
        borders = 5;
        expand_x = true;
        width = 300;
        expand_y = true;
        resizable = true;
        block
        {
            title = "Settings";
            drawbox = true;
            inplace_buttons = true;
            expand_x = true;
            expand_y = false;
            invisible dbase
            {
                value = " ";
            }

            combobox layer
            {
                label = "Output intersection layer";
                mandatory = true;
                nospaces = true;
                uppercase = true;
                input_re = "^[\p{L}0-9][\p{L}0-9_+.-]*$";
                invalid_message = "{Key design.d:Error_LayerName}";
                tooltip = "Layer to save the polygon.";
                width = 25;
                filter = "explorer(design_databases,$(dbase),(.*))";
            }

            block
            {
                numberbox radius_1
                {
                    label = "Zone 1 radius (e.g. personnel)";
                    initial = 500;
                    tooltip = "Enter radius to build polygon around blast";
                    width = 6;
                    min = 10;
                    decimal = 3;
                    value_as_double = 1;
                    invalid_message = "Radius must be greater than 10";
                }
                colour color1
                {
                    label = "";
                    position = "same line";
                }
                linestyle line1
                {
                    label = "";
                    position = "same line";
                }
                numberbox radius_2
                {
                    label = "Zone 2 radius (e.g. equipment)";
                    initial = 500;
                    tooltip = "Enter radius to build polygon around blast";
                    width = 6;
                    min = 10;
                    decimal = 3;
                    value_as_double = 1;
                    invalid_message = "Radius must be greater than 10";
                }
                colour color2
                {
                    label = "";
                    position = "same line";
                }
                linestyle line2
                {
                    label = "";
                    position = "same line";
                }
            }
        }
        block
        {
            title = "Method";
            drawbox = true;
            expand_x = true;
            expand_y = true;
            resizable = true;
            borders = 5;

            radiobutton twod
            {
                label = "Horizontal radius projection";
                bank=1;
            }
            radiobutton threed
            {
                label = "Use surface for radius projection";
                bank=1;
            }

            checkbox register
            {
                label = "Register lines to surface";
                *bank=1;
            }

            fileselector topo
            {
                label = "Topography triangulation";
                mandatory = true;
                enable = "($(threed)=1)|($(register)=1)";
                width = 32;
                tooltip = "Surface to calculate the influence polygon for";
                wildcard = "Triangulation file(*.??t)|*.??t";
                filter = "__filelist(,*.??t)";
                resizable = true;
            }
        }
        button ok
        {
            label = "OK";
        }
        button cancel
        {
            label = "Cancel";
        }
    }
    '''

    # Compile the gpan file into a cgp file
    basename = "Flyrock_polygon"

    cgp_file = Path(os.environ["TEMP"], basename + ".cgp")
    gpan_file = Path(os.environ["TEMP"], basename + ".gpan")
    cgp_file.unlink
    gpan_file.unlink

    gpan_file.write_text(gpan_string)
    gpan_compiler = Path(os.environ["VULCAN_EXE"], "pc.exe")

    subprocess.run([str(gpan_compiler), str(gpan_file)], check=True)
    if not cgp_file.exists() or cgp_file.stat().st_size <= 0:
        sys.exit("Failed to compile " + str(gpan_file))

    # Populating an invisible field with the dgd name to enable layer filtering
    spec = basename + ".spec"
    var = vulcan.variant()
    if os.path.exists(spec):
        var.load(spec)

    var["dbase"] = vulcan_gui.globals()["dgd"]
    var.save_as(spec)
    panel = vulcan_gui.gpan()
    panel.load_spec(spec)
    action = panel.display(str(cgp_file), "Main")
    if action != "ok":
        exit()
    panel.save_spec(spec)

    # Remove the generated gpan files
    remove(gpan_file)
    remove(cgp_file)

    # Define the projection method 2D / 3D
    values = panel.get_values()
    values["mode"] = 0
    if values["twod"] == 1:
        values["mode"] = 1


    return values


def create_obj_legacy(point_list, layer_name, color, line):
    r"""
    Finds the endpoint that meets the specified 3D distance with recursion and
    the topographic triangulation.

    Parameters
    ----------
    point_list (int):
    layer_name (str):
    color (int)
    line (list)

    Returns
    ----------
    None
    """

    dgd_file = vulcan_gui.globals()["dgd"]
    dgd = vulcan.dgd(dgd_file, "w")
    layer = vulcan.layer(layer_name)

    for poly in point_list:
        obj = vulcan.polyline(poly)
        obj.set_connected()
        obj.set_colour(color)
        obj.set_linetype(line)
        obj.set_closed()
        layer.append(obj)

    dgd.save_layer(layer)
    dgd.close()
    vulcan_gui.add_layer(layer)


def create_obj(point_list, pan):
    r"""
    Finds the endpoint that meets the specified 3D distance with recursion and
    the topographic triangulation.

    Parameters
    ----------
    point_list (int):
    layer_name (str):
    color (int)
    line (list)

    Returns
    ----------
    None
    """

    dgd_file = vulcan_gui.globals()["dgd"]
    dgd = vulcan.dgd(dgd_file, "w")
    layer = vulcan.layer(pan["layer"])

    colors = [pan["color1"], pan["color2"]]
    lines = [pan["line1"], pan["line2"]]

    for i in range(2):
        obj = vulcan.polyline(point_list[i])
        obj.set_connected()
        obj.set_colour(colors[i])
        obj.set_linetype(lines[i])
        obj.set_closed()
        layer.append(obj)

    dgd.save_layer(layer)
    dgd.close()
    vulcan_gui.add_layer(layer)


def register(reg, topography, layer_name):
    r"""
    Finds the endpoint that meets the specified 3D distance with recursion and
    the topographic triangulation.

    Parameters
    ----------
    reg (int): 1= yes    0 = No.
    topography (str): Triangulation path.
    layer_name (str):

    Returns
    ----------
    None
    """

    # Exit if registration is unchecked
    if reg == 0:
        return

    vulcan_gui.load_triangulation(topography)

    with run_menu("DESIGN_OBJ_REG","abort", err=None) as macro:
        macro.triangulation(topography)
        data = {
            "HighLow" : 0,
            "HighestPoint" : 0,
            "Interpolate" : 1,
            "LowestPoint" : 1,
            "button_pressed" : "ok",
            "by_2D" : 1,
            "by_3D" : 0,
            "by_object" : 1,
            "by_point" : 0,
            "by_segment" : 0,
            "plane_a" : 0.0,
            "plane_b" : 0.0,
            "plane_bearing" : 0.0,
            "plane_c" : 1.0,
            "plane_d" : 0.0,
            "plane_dip" : 90.0,
            "plane_easting" : 0.0,
            "plane_elevation" : 0.0,
            "plane_method" : "Level",
            "plane_northing" : 0.0,
            "plane_origin_x" : 0.0,
            "plane_origin_y" : 0.0,
            "plane_origin_z" : 0.0,
            "plane_plunge" : 0.0,
            "plane_strike" : 0.0,
            "surfaceIsTriangulation" : 1,
            "use_projection_plane" : 0
        }
        macro.panel_results("StringRegister", data)
        macro.select_by_attr(layer=layer_name,object="*",group="*",feature="*")
        macro.command("SELECT_CANCEL")
        macro.command("FINISH")


def main():
    print("Safety_Polygon_v1.py")

    # Get selection
    points = selection()

    # Display panel
    vals = gpan()
    vulcan_gui.start_busy_message("Processing. Please wait...")

    # Create flyrock polygons
    start = time.time()
    poly_1 = create_points(vals["topo"], vals["radius_1"], points, vals["mode"])
    poly_2 = create_points(vals["topo"], vals["radius_2"], points, vals["mode"])

    # Create objects
    create_obj([poly_1, poly_2], vals)

    # Register object to the topography
    register(vals["register"], vals["topo"], vals["layer"])

    # Report duration
    duration = time.time() - start
    print(f"Duration: {duration} seconds")

    vulcan_gui.end_busy_message()


if __name__ == "__main__":
    main()
