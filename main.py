import logging
import os
import sys
import subprocess
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import argparse

# 添加日志，设置日志级别为error
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.ERROR)
def get_git_changes(repo_path):
    """获取Git仓库中的更改内容"""
    os.chdir(repo_path)
    # 检查是否是git仓库
    if not os.path.exists(os.path.join(repo_path, '.git')):
        print(f"错误: {repo_path} 不是一个有效的Git仓库")
        sys.exit(1)

    # 获取未提交的更改
    try:
        diff_output = subprocess.check_output(['git', 'diff', '--cached', '--name-status']).decode('utf-8')
        if not diff_output.strip():
            # 如果没有已暂存的更改，获取所有未暂存的更改
            diff_output = subprocess.check_output(['git', 'diff', '--name-status']).decode('utf-8')
            if not diff_output.strip():
                print("没有检测到任何更改，无需提交")
                sys.exit(0)
            print("检测到未暂存的更改，正在自动暂存...")
            subprocess.run(['git', 'add', '.'], check=True)

        # 获取详细的更改内容
        diff_detail = subprocess.check_output(['git', 'diff', '--cached']).decode('utf-8')
        return diff_output, diff_detail
    except subprocess.CalledProcessError as e:
        print(f"执行Git命令时出错: {e}")
        sys.exit(1)


def generate_commit_message(diff_output, diff_detail, model_name="qwen2.5:3b"):
    """使用Ollama生成提交信息"""
    try:
        # 初始化Ollama
        llm = OllamaLLM(model=model_name)

        # 创建提示模板
        template = """
        作为一个Git提交信息生成器，请根据以下Git更改生成一个简洁、清晰且符合约定式提交规范的提交信息。

        更改的文件列表:
        {diff_output}

        详细更改内容:
        {diff_detail}

        请生成一个包含类型前缀的简短提交消息（如feat:, fix:, docs:, style:, refactor:, perf:, test:, build:, ci:, chore:）。
        提交信息应当简洁（不超过100个字符），并且应该说明这次提交的内容和原因。

        提交信息:
        """

        # 创建提示
        prompt = PromptTemplate(
            template=template,
            input_variables=["diff_output", "diff_detail"]
        )

        # 创建链
        chain = LLMChain(llm=llm, prompt=prompt)

        # 运行链并生成提交信息
        commit_message = chain.run(diff_output=diff_output, diff_detail=diff_detail[:4000])  # 限制输入大小

        # 清理输出
        commit_message = commit_message.strip().replace('\n', ' ')
        # 确保不超过100个字符
        if len(commit_message) > 100:
            commit_message = commit_message[:97] + "..."

        return commit_message
    except Exception as e:
        print(f"生成提交信息时出错: {e}")
        sys.exit(1)


def commit_and_push(repo_path, commit_message, remote="origin", branch="main"):
    """提交更改并推送到远程仓库"""
    os.chdir(repo_path)
    try:
        # 提交更改
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        print(f"已成功提交: {commit_message}")

        # 检查远程仓库是否存在
        remote_exists = subprocess.run(['git', 'remote', 'get-url', remote],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE).returncode == 0

        if remote_exists:
            # 获取当前分支(如果未指定)
            if branch == "main":
                try:
                    branch = subprocess.check_output(['git', 'branch', '--show-current']).decode('utf-8').strip()
                    if not branch:
                        branch = "main"  # 使用默认分支
                except:
                    branch = "main"

            # 推送到远程仓库
            print(f"正在推送到远程仓库 {remote}/{branch}...")
            subprocess.run(['git', 'push', remote, branch], check=True)
            print(f"已成功推送到 {remote}/{branch}")
        else:
            print(f"警告: 未找到远程仓库 '{remote}'，跳过推送操作")

    except subprocess.CalledProcessError as e:
        print(f"提交或推送时出错: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='自动生成Git提交信息并推送到远程仓库')
    parser.add_argument('--repo', type=str, default=os.getcwd(),
                        help='Git仓库的路径 (默认为当前目录)')
    parser.add_argument('--model', type=str, default='qwen2.5:3b',
                        help='要使用的Ollama模型名称 (默认为llama3)')
    parser.add_argument('--remote', type=str, default='origin',
                        help='远程仓库名称 (默认为origin)')
    parser.add_argument('--branch', type=str, default='main',
                        help='分支名称 (默认为当前分支，如果无法确定则使用main)')
    parser.add_argument('--no-push', action='store_true',
                        help='仅提交更改，不推送到远程仓库')

    args = parser.parse_args()

    # 获取Git更改
    diff_output, diff_detail = get_git_changes(args.repo)

    # 生成提交信息
    commit_message = generate_commit_message(diff_output, diff_detail, args.model)
    print(f"生成的提交信息: {commit_message}")

    # 提交并推送
    if args.no_push:
        commit_and_push(args.repo, commit_message, None, None)
    else:
        commit_and_push(args.repo, commit_message, args.remote, args.branch)


if __name__ == "__main__":
    main()