import argparse
import hashlib
import os
import subprocess
import sys
import time

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema


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


def get_current_branch():
    """获取当前分支名称"""
    try:
        branch = subprocess.check_output(['git', 'branch', '--show-current']).decode('utf-8').strip()
        return branch if branch else "main"
    except:
        return "main"


def setup_milvus_lite(uri='.milvus.db', collection_name='git_commits'):
    """设置MilvusLite连接并创建集合（如果不存在）"""
    try:
        # 连接到MilvusLite
        client = MilvusClient(uri=uri)
        model = OllamaLLM(model="qwen2.5:3b")
        embeddings_model = OllamaEmbeddings(model="qwen2.5:3b")
        # 检查集合是否存在
        collections = client.list_collections()
        if collection_name not in collections:

            fields = [
                FieldSchema(name="commit_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, description="ID of the commit", max_length= 100),
                FieldSchema(name="repo_path", dtype=DataType.VARCHAR, description="Path to the git repository", max_length= 500),
                FieldSchema(name="branch", dtype=DataType.VARCHAR, max_length=100, description="Branch name",),
                FieldSchema(name="commit_message", dtype=DataType.VARCHAR, max_length=500, description="The commit message",),
                FieldSchema(name="commit_timestamp", dtype=DataType.INT64, description="Timestamp of the commit",),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="Vector embedding of commit message", dim= 384 ),
            ]

            schema = CollectionSchema(fields=fields, enable_dynamic_field=True,)
            # 创建集合
            # schema = {
            #     "fields": [
            #         {
            #             "name": "commit_id",
            #             "description": "ID of the commit",
            #             "data_type": "varchar",
            #             "max_length": 100,
            #             "is_primary": True,
            #         },
            #         {
            #             "name": "repo_path",
            #             "description": "Path to the git repository",
            #             "data_type": "varchar",
            #             "max_length": 500,
            #         },
            #         {
            #             "name": "branch",
            #             "description": "Branch name",
            #             "data_type": "varchar",
            #             "max_length": 100,
            #         },
            #         {
            #             "name": "commit_message",
            #             "description": "The commit message",
            #             "data_type": "varchar",
            #             "max_length": 500,
            #         },
            #         {
            #             "name": "commit_timestamp",
            #             "description": "Timestamp of the commit",
            #             "data_type": "int64",
            #         },
            #         {
            #             "name": "embedding",
            #             "description": "Vector embedding of commit message",
            #             "data_type": "float_vector",
            #             "dim": 384  # 使用Ollama的向量维度
            #         }
            #     ],
            #     "enable_dynamic_field": True
            # }

            client.create_collection(
                collection_name=collection_name,
                schema=schema
            )
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                metric_type="COSINE",
                index_type="IVF_FLAT",
                index_name="vector_index",
                params={"M": 8, "efConstruction": 64}
            )

            client.create_index(
                collection_name=collection_name,
                field_name="embedding",
                index_params=index_params,
            )

            print(f"创建集合 '{collection_name}' 成功")
        else:
            print(f"集合 '{collection_name}' 已存在")

        return client
    except Exception as e:
        print(f"设置MilvusLite连接时出错: {e}")
        return None


def get_recent_commits_from_milvus(client, repo_path, branch, collection_name='git_commits', limit=5):
    """从MilvusLite获取特定分支的最近commit信息"""
    try:
        # 查询特定仓库和分支的最近提交
        res = client.query(
            collection_name=collection_name,
            filter=f'repo_path == "{repo_path}" and branch == "{branch}"',
            output_fields=["commit_message", "commit_timestamp"],
            limit=limit
        )

        # 按时间戳排序（因为Milvus查询没有自带排序）
        results = sorted(res, key=lambda x: x["commit_timestamp"], reverse=True)

        return [result["commit_message"] for result in results]
    except Exception as e:
        print(f"从MilvusLite获取commit信息时出错: {e}")
        return []


def save_commit_to_milvus(client, repo_path, branch, commit_message, embedding, collection_name='git_commits'):
    """将commit信息保存到MilvusLite"""
    try:
        # 生成唯一ID
        commit_id = hashlib.md5(f"{repo_path}_{branch}_{commit_message}_{time.time()}".encode()).hexdigest()

        # 准备要插入的数据
        data = {
            "commit_id": commit_id,
            "repo_path": repo_path,
            "branch": branch,
            "commit_message": commit_message,
            "commit_timestamp": int(time.time()),
            "embedding": embedding
        }

        # 插入数据
        client.insert(
            collection_name=collection_name,
            data=[data]
        )

        print(f"成功将commit信息保存到MilvusLite")
        return True
    except Exception as e:
        print(f"保存commit信息到MilvusLite时出错: {e}")
        return False


def generate_commit_message(diff_output, diff_detail, client, repo_path, branch, model_name="qwen2.5:3b",
                            collection_name='git_commits'):
    """使用Ollama生成提交信息，并参考之前的commit历史"""
    try:
        # 初始化Ollama
        llm = OllamaLLM(model=model_name)
        embeddings_model = OllamaEmbeddings(model=model_name)

        # 获取该分支最近的提交历史
        recent_commits = get_recent_commits_from_milvus(client, repo_path, branch, collection_name)
        recent_commits_str = "\n".join([f"- {commit}" for commit in recent_commits]) if recent_commits else "无历史提交记录"

        # 创建提示模板
        template = """
        作为一个Git提交信息生成器，请根据以下Git更改生成一个简洁、清晰且符合约定式提交规范的提交信息。

        更改的文件列表:
        {diff_output}

        详细更改内容:
        {diff_detail}

        该分支的最近提交历史:
        {recent_commits}

        请生成一个包含类型前缀的简短提交消息（如feat:, fix:, docs:, style:, refactor:, perf:, test:, build:, ci:, chore:）。
        提交信息应当简洁（不超过100个字符），并且应该说明这次提交的内容和原因。
        参考最近的提交历史，保持风格一致并避免重复。

        提交信息:
        """

        # 创建提示
        prompt = PromptTemplate(
            template=template,
            input_variables=["diff_output", "diff_detail", "recent_commits"]
        )

        # 创建链
        chain = LLMChain(llm=llm, prompt=prompt)

        # 运行链并生成提交信息
        commit_message = chain.run(
            diff_output=diff_output,
            diff_detail=diff_detail[:4000],  # 限制输入大小
            recent_commits=recent_commits_str
        )

        # 清理输出
        commit_message = commit_message.strip().replace('\n', ' ')
        # 确保不超过100个字符
        if len(commit_message) > 100:
            commit_message = commit_message[:97] + "..."

        return commit_message, embeddings_model
    except Exception as e:
        print(f"生成提交信息时出错: {e}")
        sys.exit(1)


def commit_and_push(repo_path, commit_message, client, embeddings_model, remote="origin", branch="main",
                    collection_name='git_commits'):
    """提交更改并推送到远程仓库，同时保存commit信息到MilvusLite"""
    os.chdir(repo_path)
    try:
        # 提交更改
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        print(f"已成功提交: {commit_message}")

        # 获取嵌入向量并保存commit信息到MilvusLite
        embedding = embeddings_model.embed_query(commit_message)
        save_commit_to_milvus(client, repo_path, branch, commit_message, embedding, collection_name)

        # 检查远程仓库是否存在
        if remote:
            remote_exists = subprocess.run(['git', 'remote', 'get-url', remote],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE).returncode == 0

            if remote_exists:
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
    parser.add_argument('--branch', type=str, default=None,
                        help='分支名称 (默认为当前分支)')
    parser.add_argument('--no-push', action='store_true',
                        help='仅提交更改，不推送到远程仓库')
    parser.add_argument('--milvus-uri', type=str, default='milvus.db',
                        help='MilvusLite URI (默认为当前目录下的milvus.db)')
    parser.add_argument('--collection', type=str, default='git_commits',
                        help='Milvus集合名称 (默认为git_commits)')

    args = parser.parse_args()

    # 使用指定分支或获取当前分支
    branch = args.branch if args.branch else get_current_branch()

    # 设置MilvusLite连接
    client = setup_milvus_lite(args.milvus_uri, args.collection)
    if client is None:
        print("无法连接到MilvusLite，将退出程序")
        sys.exit(1)

    # 获取Git更改
    diff_output, diff_detail = get_git_changes(args.repo)

    # 生成提交信息，同时获取嵌入模型以便后续使用
    commit_message, embeddings_model = generate_commit_message(
        diff_output, diff_detail, client, args.repo, branch, args.model, args.collection
    )
    print(f"生成的提交信息: {commit_message}")

    # 提交并推送
    if args.no_push:
        commit_and_push(args.repo, commit_message, client, embeddings_model, None, branch, args.collection)
    else:
        commit_and_push(args.repo, commit_message, client, embeddings_model, args.remote, branch, args.collection)


if __name__ == "__main__":
    main()