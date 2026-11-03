import sys
import argparse
import configparser


if __name__ == "__main__":
    print(sys.argv)
    parser = argparse.ArgumentParser(description='modify out nodes indexes of model convert ini and post process script')

    parser.add_argument('-n', '--out_nodes', type=str, help='onnx out nodes indexes', required=True)
    parser.add_argument('-i', '--ini_path', type=str, help='input model convert ini path')
    parser.add_argument('-p', '--postprocess_py', type=str, help='postprocess python script path')
    args = parser.parse_args()


    print(args)


    config_parser = configparser.ConfigParser()

    config_parser.read(args.ini_path)

    out_nodes = args.out_nodes.split(",")
    print("out_nodes", out_nodes)
    print(",".join(list(map(str, out_nodes))))

    config_parser.set("OUTPUT_CONFIG", "outputs", ",".join(list(map(str, out_nodes))))
    config_parser.set("OUTPUT_CONFIG", "dequantizations", ",".join(["TRUE"] * len(out_nodes)))

    config_parser.write(open(args.ini_path, "w"))

    print(f"write {','.join(list(map(str, out_nodes)))}")

    py_lines = []

    with open(args.postprocess_py, "r") as f:
        py_lines = f.readlines()

    for i_line in range(len(py_lines)):
        if py_lines[i_line].endswith("# PLACEHOLDER_OUT_NODES\n"):
            py_lines[i_line] = "            '" + "','".join(out_nodes) + "' # PLACEHOLDER_OUT_NODES\n"
            break

    # print("".join(py_lines))

    with open(args.postprocess_py, "w") as f:
        f.write("".join(py_lines))