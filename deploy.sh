#!/bin/bash

# Memsys Docker 部署脚本
# Memory System Docker Deployment Script

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_success() {
    print_message $GREEN "✅ $1"
}

print_warning() {
    print_message $YELLOW "⚠️  $1"
}

print_error() {
    print_message $RED "❌ $1"
}

print_info() {
    print_message $BLUE "ℹ️  $1"
}

# 检查 Docker 和 Docker Compose
check_dependencies() {
    print_info "检查依赖..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi
    
    print_success "依赖检查通过"
}

# 检查环境配置文件
check_env_file() {
    print_info "检查环境配置..."
    
    if [ ! -f ".env" ]; then
        print_warning ".env 文件不存在，正在创建..."
        cp docker.env .env
        print_warning "请编辑 .env 文件，配置您的 API 密钥"
        print_warning "特别是以下配置项："
        print_warning "  - CONV_MEMCELL_LLM_API_KEY"
        print_warning "  - EPISODE_MEMORY_LLM_API_KEY"
        print_warning "  - DEEPINFRA_API_KEY (可选)"
        print_warning "  - SILICONFLOW_API_KEY (可选)"
        
        read -p "是否现在编辑 .env 文件? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ${EDITOR:-nano} .env
        fi
    fi
    
    print_success "环境配置检查完成"
}

# 创建必要的目录
create_directories() {
    print_info "创建必要的目录..."
    
    mkdir -p data logs docker/mongodb/init
    
    print_success "目录创建完成"
}

# 启动依赖服务（应用本地运行）
start_services() {
    print_info "启动依赖服务..."
    
    # 启动依赖服务（不包含应用，不包含 PostgreSQL）
    docker-compose up -d mongodb elasticsearch milvus-etcd milvus-minio milvus-standalone redis
    
    print_success "依赖服务启动完成"
}

# 等待服务就绪
wait_for_services() {
    print_info "等待服务就绪..."
    
    # 等待 MongoDB
    print_info "等待 MongoDB..."
    timeout 60 bash -c 'until docker-compose exec -T mongodb mongosh --eval "db.adminCommand(\"ping\")" > /dev/null 2>&1; do sleep 2; done'
    
    # 等待 Elasticsearch
    print_info "等待 Elasticsearch..."
    timeout 60 bash -c 'until curl -f http://localhost:9200/_cluster/health > /dev/null 2>&1; do sleep 2; done'
    
    # 等待 Redis
    print_info "等待 Redis..."
    timeout 30 bash -c 'until docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; do sleep 2; done'
    
    # 等待 Milvus
    print_info "等待 Milvus..."
    timeout 120 bash -c 'until curl -f http://localhost:9091/healthz > /dev/null 2>&1; do sleep 5; done'
    
    print_success "所有服务已就绪"
}

# 显示服务状态
show_status() {
    print_info "服务状态："
    docker-compose ps
    
    echo
    print_info "服务访问地址："
    print_info "  - 本地运行的 Memsys 应用: http://localhost:1995"
    print_info "  - Elasticsearch: http://localhost:9200"
    print_info "  - Milvus MinIO 控制台: http://localhost:9001"
    print_info "  - MongoDB: localhost:27017"
    print_info "  - Redis: localhost:6379"
}

# 显示日志
show_logs() {
    print_info "应用在本地运行，请使用本地日志方式查看。例如："
    print_info "  uv run python run.py"
}

# 停止服务
stop_services() {
    print_info "停止服务..."
    docker-compose down
    print_success "服务已停止"
}

# 清理数据
clean_data() {
    print_warning "这将删除所有数据，包括数据库、缓存和日志！"
    read -p "确定要继续吗? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "清理数据..."
        docker-compose down -v
        docker system prune -f
        print_success "数据清理完成"
    else
        print_info "取消清理操作"
    fi
}

# 显示帮助信息
show_help() {
    echo "Memsys Docker 部署脚本"
    echo
    echo "用法: $0 [命令]"
    echo
    echo "命令:"
    echo "  start     启动所有服务 (默认)"
    echo "  stop      停止所有服务"
    echo "  restart   重启所有服务"
    echo "  status    显示服务状态"
    echo "  logs      显示应用日志"
    echo "  clean     清理所有数据"
    echo "  help      显示帮助信息"
    echo
    echo "示例:"
    echo "  $0 start    # 启动服务"
    echo "  $0 logs     # 查看日志"
    echo "  $0 stop     # 停止服务"
}

# 主函数
main() {
    local command=${1:-start}
    
    case $command in
        start)
            check_dependencies
            check_env_file
            create_directories
            start_services
            wait_for_services
            show_status
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 2
            check_dependencies
            check_env_file
            create_directories
            start_services
            wait_for_services
            show_status
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        clean)
            clean_data
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"

