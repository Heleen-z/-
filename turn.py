"""
批量图片格式转换脚本
将PNG、BMP、TIFF等格式转换为JPG
"""

from PIL import Image
import os
import sys

def convert_to_jpg(input_folder, output_folder, quality=85):
    """
    批量转换图片为JPG格式
    
    参数:
    input_folder: 输入文件夹路径
    output_folder: 输出文件夹路径
    quality: JPG质量 (1-100)，默认85
    """
    
    # 1. 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"❌ 错误：输入文件夹 '{input_folder}' 不存在！")
        return False
    
    # 2. 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 3. 支持的输入格式
    supported_formats = ('.png', '.bmp', '.tiff', '.tif', '.gif', 
                         '.webp', '.jpeg', '.jpg', '.ppm', '.ico')
    
    # 4. 统计变量
    total_files = 0
    converted_files = 0
    failed_files = []
    
    print("=" * 50)
    print(f"📁 输入文件夹: {input_folder}")
    print(f"📁 输出文件夹: {output_folder}")
    print("=" * 50)
    
    # 5. 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件扩展名
        if filename.lower().endswith(supported_formats):
            total_files += 1
            
            try:
                # 构建完整文件路径
                input_path = os.path.join(input_folder, filename)
                
                # 获取文件名（不含扩展名）
                name, ext = os.path.splitext(filename)
                
                # 输出文件名
                output_filename = f"{name}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                
                # 打开图片
                with Image.open(input_path) as img:
                    # 转换为RGB模式（JPG不支持RGBA）
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # 创建一个白色背景
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        rgb_img = background
                    else:
                        rgb_img = img.convert('RGB')
                    
                    # 保存为JPG
                    rgb_img.save(
                        output_path,
                        'JPEG',
                        quality=quality,
                        optimize=True
                    )
                    
                    converted_files += 1
                    print(f"✅ 已转换: {filename} → {output_filename}")
                    
            except Exception as e:
                failed_files.append((filename, str(e)))
                print(f"❌ 转换失败: {filename} - {str(e)}")
    
    # 6. 显示统计结果
    print("\n" + "=" * 50)
    print("📊 转换完成！")
    print(f"总计文件: {total_files}")
    print(f"成功转换: {converted_files}")
    
    if failed_files:
        print(f"失败文件: {len(failed_files)}")
        print("失败详情:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    print(f"所有文件已保存到: {output_folder}")
    print("=" * 50)
    
    return True

def main():
    """主函数：处理用户交互"""
    
    print("🖼️ 批量图片格式转换工具 (Python)")
    print("功能：将PNG、BMP、TIFF等格式批量转换为JPG")
    print("-" * 50)
    
    # 设置默认路径（当前目录下的文件夹）
    current_dir = os.getcwd()
    default_input = os.path.join(current_dir, "input_images")
    default_output = os.path.join(current_dir, "output_images")
    
    # 获取用户输入
    print(f"当前目录: {current_dir}")
    
    # 1. 输入文件夹路径
    input_folder = input(f"请输入输入文件夹路径 [默认: {default_input}]：").strip()
    if not input_folder:
        input_folder = default_input
    
    # 2. 输出文件夹路径
    output_folder = input(f"请输入输出文件夹路径 [默认: {default_output}]：").strip()
    if not output_folder:
        output_folder = default_output
    
    # 3. JPG质量设置
    quality_input = input("请输入JPG质量 (1-100，默认85)：").strip()
    try:
        quality = int(quality_input) if quality_input else 85
        if quality < 1 or quality > 100:
            print("⚠️  质量值应在1-100之间，使用默认值85")
            quality = 85
    except ValueError:
        print("⚠️  输入无效，使用默认质量85")
        quality = 85
    
    # 4. 确认信息
    print("\n" + "=" * 50)
    print("请确认以下设置：")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"JPG质量: {quality}")
    print("=" * 50)
    
    confirm = input("是否开始转换？(y/n): ").strip().lower()
    
    if confirm == 'y' or confirm == 'yes' or confirm == '':
        print("🚀 开始转换...")
        convert_to_jpg(input_folder, output_folder, quality)
        
        # 转换完成后暂停（Windows）
        if sys.platform == 'win32':
            input("\n按Enter键退出...")
    else:
        print("❌ 已取消转换")
        
        if sys.platform == 'win32':
            input("按Enter键退出...")

if __name__ == "__main__":
    main()