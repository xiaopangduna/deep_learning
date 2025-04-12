
# import torch
# import time
# import os
# from torchsummary import summary
# from thop import profile
# import sys


# class LayerTimeProfiler:
#     def __init__(self, model, use_gpu=False, warmup_runs=3, actual_runs=10, save_folder="profiling_results"):
#         self.model = model
#         self.use_gpu = use_gpu
#         self.warmup_runs = warmup_runs
#         self.actual_runs = actual_runs
#         self.save_folder = save_folder
#         self.layer_times = {}
#         self.total_time = 0
#         self.model_summary = None
#         self.total_params = 0
#         self.flops = 0

#     def profile(self, input_tensor):
#         self.model.eval()
#         self.model = self.model.cuda()
#         input_tensor=input_tensor.cuda()
#         # 热身运行
#         for _ in range(self.warmup_runs):
#             with torch.no_grad():
#                 _ = self.model(input_tensor)

#         layer_run_times = {}
#         total_run_times = []

#         for _ in range(self.actual_runs):
#             if self.use_gpu:
#                 torch.cuda.synchronize()
#                 start_event = torch.cuda.Event(enable_timing=True)
#                 end_event = torch.cuda.Event(enable_timing=True)
#                 start_event.record()
#             else:
#                 start_time = time.time()

#             with torch.no_grad():
#                 temp_input = input_tensor.clone()
#                 for name, layer in self.model.named_children():
#                     if self.use_gpu:
#                         layer_start_event = torch.cuda.Event(enable_timing=True)
#                         layer_end_event = torch.cuda.Event(enable_timing=True)
#                         layer_start_event.record()
#                     else:
#                         layer_start_time = time.time()

#                     temp_input = layer(temp_input)

#                     if self.use_gpu:
#                         layer_end_event.record()
#                         torch.cuda.synchronize()
#                         layer_elapsed_time = layer_start_event.elapsed_time(layer_end_event)
#                     else:
#                         layer_end_time = time.time()
#                         layer_elapsed_time = (layer_end_time - layer_start_time) * 1000

#                     if name not in layer_run_times:
#                         layer_run_times[name] = []
#                     layer_run_times[name].append(layer_elapsed_time)

#             if self.use_gpu:
#                 end_event.record()
#                 torch.cuda.synchronize()
#                 total_elapsed_time = start_event.elapsed_time(end_event)
#             else:
#                 end_time = time.time()
#                 total_elapsed_time = (end_time - start_time) * 1000

#             total_run_times.append(total_elapsed_time)

#         # 计算每层的最大、最小和平均运行时间
#         for name, times in layer_run_times.items():
#             self.layer_times[name] = {
#                 "max": max(times),
#                 "min": min(times),
#                 "avg": sum(times) / len(times)
#             }

#         # 计算整个模型的平均运行时间
#         self.total_time = sum(total_run_times) / len(total_run_times)

#         # # 计算参数量

#         from io import StringIO
#         old_stdout = sys.stdout
#         sys.stdout = mystdout = StringIO()
#         summary(self.model, input_size=tuple(input_tensor.shape[1:]), device="cuda")
#         self.model_summary = mystdout.getvalue()
#         sys.stdout = old_stdout
#         self.total_params = sum(p.numel() for p in self.model.parameters())

#         # 计算计算量
#         self.flops, _ = profile(self.model, inputs=(input_tensor,))

#         return input_tensor

#     def get_sorted_times(self):
#         sorted_layer_times = sorted(self.layer_times.items(), key=lambda item: item[1]["avg"], reverse=True)
#         return [("Total", {
#             "max": self.total_time,
#             "min": self.total_time,
#             "avg": self.total_time
#         })] + sorted_layer_times

#     def generate_report(self, model_name, save_to_file=True):
#         report = f"模型: {model_name}\n"
#         report += "+---------------------+---------------------+\n"
#         report += "| 指标                | 数值                |\n"
#         report += "+---------------------+---------------------+\n"
#         report += f"| 总参数量            | {self.total_params:<19}|\n"
#         report += f"| 总计算量 (FLOPs)    | {self.flops:<19}|\n"
#         report += f"| 总运行时间 (毫秒)   | {self.total_time:<19.6f}|\n"
#         report += "+---------------------+---------------------+\n"

#         # report += "\n模型结构：\n"
#         # report += self.model_summary

#         report += "\n每一层运行时间（按平均时间从大到小排序）：\n"
#         report += "+---------------------+---------------------+---------------------+---------------------+\n"
#         report += "| 层名称              | 最大运行时间 (毫秒) | 最小运行时间 (毫秒) | 平均运行时间 (毫秒) |\n"
#         report += "+---------------------+---------------------+---------------------+---------------------+\n"
#         for layer_name, times in self.get_sorted_times():
#             report += f"| {layer_name.ljust(19)} | {times['max']:<19.6f} | {times['min']:<19.6f} | {times['avg']:<19.6f} |\n"
#         report += "+---------------------+---------------------+---------------------+---------------------+\n"

#         if save_to_file:
#             if not os.path.exists(self.save_folder):
#                 os.makedirs(self.save_folder)
#             file_path = os.path.join(self.save_folder, f"{model_name}_report.txt")
#             with open(file_path, "w") as f:
#                 f.write(report)
#             print(f"{model_name} 的详细报告已保存到 {file_path}")
#         return report


# def compare_models(models, input_tensor, use_gpu=False, warmup_runs=3, actual_runs=10, save_folder="profiling_results"):
#     all_results = {}
#     for model_name, model in models.items():
#         try:
#             if use_gpu:
#                 model = model.cuda()

#             profiler = LayerTimeProfiler(model, use_gpu=use_gpu, warmup_runs=warmup_runs, actual_runs=actual_runs,
#                                          save_folder=save_folder)
#             profiler.profile(input_tensor)
#             profiler.generate_report(model_name)

#             all_results[model_name] = {
#                 "total_time": profiler.total_time,
#                 "params": profiler.total_params,
#                 "flops": profiler.flops
#             }
#         except Exception as e:
#             print(f"分析模型 {model_name} 时出错: {e}")

#     # 生成对比表格
#     table_header = "+---------------------+---------------------+---------------------+---------------------+\n"
#     table_header += "| 模型名称            | 总运行时间 (毫秒)   | 总参数量            | 总计算量 (FLOPs)    |\n"
#     table_header += "+---------------------+---------------------+---------------------+---------------------+\n"
#     table_content = ""
#     for model_name, results in all_results.items():
#         table_content += f"| {model_name.ljust(19)} | {results['total_time']:<19.6f} | {results['params']:<19} | {results['flops']:<19} |\n"
#     table_footer = "+---------------------+---------------------+---------------------+---------------------+\n"

#     comparison_report = table_header + table_content + table_footer
#     print("\n多模型对比表格：")
#     print(comparison_report)

#     # 保存对比报告
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     file_path = os.path.join(save_folder, f"comparison_report_{timestamp}.txt")
#     with open(file_path, "w") as f:
#         f.write(comparison_report)
#     print(f"对比报告已保存到 {file_path}")
import torch
import time
import os
from torchsummary import summary
from thop import profile
import sys


class LayerTimeProfiler:
    def __init__(self, model, use_gpu=False, warmup_runs=3, actual_runs=10, save_folder="profiling_results"):
        self.model = model
        self.use_gpu = use_gpu
        self.warmup_runs = warmup_runs
        self.actual_runs = actual_runs
        self.save_folder = save_folder
        self.layer_times = {}
        self.total_time = 0
        self.model_summary = None
        self.total_params = 0
        self.flops = 0

    def profile_layer(self, layer, input_tensor):
        if self.use_gpu:
            layer_start_event = torch.cuda.Event(enable_timing=True)
            layer_end_event = torch.cuda.Event(enable_timing=True)
            layer_start_event.record()
        else:
            layer_start_time = time.time()

        output = layer(input_tensor)

        if self.use_gpu:
            layer_end_event.record()
            torch.cuda.synchronize()
            layer_elapsed_time = layer_start_event.elapsed_time(layer_end_event)
        else:
            layer_end_time = time.time()
            layer_elapsed_time = (layer_end_time - layer_start_time) * 1000

        return output, layer_elapsed_time

    def recursive_profile(self, module, input_tensor, prefix=""):
        if len(list(module.children())) == 0:
            name = prefix if prefix else str(module.__class__.__name__)
            output, elapsed_time = self.profile_layer(module, input_tensor)
            if name not in self.layer_times:
                self.layer_times[name] = []
            self.layer_times[name].append(elapsed_time)
            return output
        else:
            for sub_name, sub_module in module.named_children():
                new_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
                input_tensor = self.recursive_profile(sub_module, input_tensor, new_prefix)
            return input_tensor

    def profile(self, input_tensor):
        self.model.eval()
        if self.use_gpu:
            self.model = self.model.cuda()
            input_tensor = input_tensor.cuda()

        # 热身运行
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = self.model(input_tensor)

        total_run_times = []

        for _ in range(self.actual_runs):
            if self.use_gpu:
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()

            with torch.no_grad():
                temp_input = input_tensor.clone()
                self.recursive_profile(self.model, temp_input)

            if self.use_gpu:
                end_event.record()
                torch.cuda.synchronize()
                total_elapsed_time = start_event.elapsed_time(end_event)
            else:
                end_time = time.time()
                total_elapsed_time = (end_time - start_time) * 1000

            total_run_times.append(total_elapsed_time)

        # 计算每层的最大、最小和平均运行时间
        for name, times in self.layer_times.items():
            self.layer_times[name] = {
                "max": max(times),
                "min": min(times),
                "avg": sum(times) / len(times)
            }

        # 计算整个模型的平均运行时间
        self.total_time = sum(total_run_times) / len(total_run_times)

        # 计算参数量
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        summary(self.model, input_size=tuple(input_tensor.shape[1:]), device="cuda" if self.use_gpu else "cpu")
        self.model_summary = mystdout.getvalue()
        sys.stdout = old_stdout
        self.total_params = sum(p.numel() for p in self.model.parameters())

        # 计算计算量
        self.flops, _ = profile(self.model, inputs=(input_tensor,))

        return input_tensor

    def get_sorted_times(self):
        sorted_layer_times = sorted(self.layer_times.items(), key=lambda item: item[1]["avg"], reverse=True)
        return [("Total", {
            "max": self.total_time,
            "min": self.total_time,
            "avg": self.total_time
        })] + sorted_layer_times

    def generate_report(self, model_name, save_to_file=True):
        report = f"模型: {model_name}\n"
        report += "+---------------------+---------------------+\n"
        report += "| 指标                | 数值                |\n"
        report += "+---------------------+---------------------+\n"
        report += f"| 总参数量            | {self.total_params:<19}|\n"
        report += f"| 总计算量 (FLOPs)    | {self.flops:<19}|\n"
        report += f"| 总运行时间 (毫秒)   | {self.total_time:<19.6f}|\n"
        report += "+---------------------+---------------------+\n"

        # report += "\n模型结构：\n"
        # report += self.model_summary

        report += "\n每一层运行时间（按平均时间从大到小排序）：\n"
        report += "+---------------------+---------------------+---------------------+---------------------+\n"
        report += "| 层名称              | 最大运行时间 (毫秒) | 最小运行时间 (毫秒) | 平均运行时间 (毫秒) |\n"
        report += "+---------------------+---------------------+---------------------+---------------------+\n"
        for layer_name, times in self.get_sorted_times():
            report += f"| {layer_name.ljust(19)} | {times['max']:<19.6f} | {times['min']:<19.6f} | {times['avg']:<19.6f} |\n"
        report += "+---------------------+---------------------+---------------------+---------------------+\n"

        if save_to_file:
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
            file_path = os.path.join(self.save_folder, f"{model_name}_report.txt")
            with open(file_path, "w") as f:
                f.write(report)
            print(f"{model_name} 的详细报告已保存到 {file_path}")
        return report


def compare_models(models, input_tensor, use_gpu=False, warmup_runs=3, actual_runs=10, save_folder="profiling_results"):
    all_results = {}
    for model_name, model in models.items():
        try:
            if use_gpu:
                model = model.cuda()

            profiler = LayerTimeProfiler(model, use_gpu=use_gpu, warmup_runs=warmup_runs, actual_runs=actual_runs,
                                         save_folder=save_folder)
            profiler.profile(input_tensor)
            profiler.generate_report(model_name)

            all_results[model_name] = {
                "total_time": profiler.total_time,
                "params": profiler.total_params,
                "flops": profiler.flops
            }
        except Exception as e:
            print(f"分析模型 {model_name} 时出错: {e}")

    # 生成对比表格
    table_header = "+---------------------+---------------------+---------------------+---------------------+\n"
    table_header += "| 模型名称            | 总运行时间 (毫秒)   | 总参数量            | 总计算量 (FLOPs)    |\n"
    table_header += "+---------------------+---------------------+---------------------+---------------------+\n"
    table_content = ""
    for model_name, results in all_results.items():
        table_content += f"| {model_name.ljust(19)} | {results['total_time']:<19.6f} | {results['params']:<19} | {results['flops']:<19} |\n"
    table_footer = "+---------------------+---------------------+---------------------+---------------------+\n"

    comparison_report = table_header + table_content + table_footer
    print("\n多模型对比表格：")
    print(comparison_report)

    # 保存对比报告
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(save_folder, f"comparison_report_{timestamp}.txt")
    with open(file_path, "w") as f:
        f.write(comparison_report)
    print(f"对比报告已保存到 {file_path}")
