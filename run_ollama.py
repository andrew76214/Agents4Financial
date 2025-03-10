import platform
import argparse
import time
import json
import re
import os
import shutil
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from example_code.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY
# 原先使用 OpenAI 的部分移除
# from openai import OpenAI
from utils import get_web_element_rect, encode_image, extract_information, print_message, \
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, clip_message_and_obs_text_only

# 改用 langchain 的 ChatOllama
from langchain_ollama import ChatOllama


def setup_logger(folder_path):
    log_file_path = os.path.join(folder_path, 'agent.log')

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def driver_config(args):
    # 設定selenium的模擬環境
    options = webdriver.ChromeOptions()

    if args.save_accessibility_tree:
        args.force_device_scale = True

    if args.force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if args.headless:
        # headless模式不會額外開啟視窗，會在背景運行任務
        options.add_argument("--headless")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    # 設定下載路徑，以便處理PDF檔案
    options.add_experimental_option(
        "prefs", {
            "download.default_directory": args.download_dir,
            "plugins.always_open_pdf_externally": True
        }
    )
    return options


def format_msg(it, init_msg, pdf_obs, warn_obs, web_img_b64, web_text):
    if it == 1:
        # 第一次執行的prompt
        init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
        init_msg_format = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': init_msg},
            ]
        }
        init_msg_format['content'].append({"type": "image_url",
                                           "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}})
        return init_msg_format
    else:
        if not pdf_obs:
            # 第二次之後，可能有錯誤訊息要附加到prompt
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ]
            }
        else:
            # 對於PDF檔案，會有另外的提示pdf_obs
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ]
            }
        return curr_msg


def format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree):
    if it == 1:
        init_msg_format = {
            'role': 'user',
            'content': init_msg + '\n' + ac_tree
        }
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': f"Observation:{warn_obs} please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree of the current page is also given, give the Thought and Action.\n{ac_tree}"
            }
        return curr_msg


def messages_to_prompt(messages):
    """
    將 messages (list of dict) 轉換成單一 prompt 字串，
    這裡將 role 與 content 依序拼接，對於附加圖片的部分，僅用 [IMAGE] 代替。
    """
    prompt = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            combined = ""
            for item in content:
                if item.get("type") == "text":
                    combined += item.get("text", "") + "\n"
                elif item.get("type") == "image_url":
                    combined += "[IMAGE]\n"
            prompt += f"{role}: {combined}\n"
        else:
            prompt += f"{role}: {content}\n"
    return prompt


def call_chatollama_api(args, llm, messages):
    """
    呼叫 ChatOllama API，將 messages 組合成 prompt 後送出，
    由於 ChatOllama 目前僅支援文字，故圖片部分以 [IMAGE] 取代。
    """
    retry_times = 0
    while True:
        try:
            prompt = messages_to_prompt(messages)
            logging.info('Calling ChatOllama API...')
            response = llm(prompt)
            # 這裡 tokens 以 0 表示，實際上可以依需求計算
            prompt_tokens = 0
            completion_tokens = 0
            call_error = False
            return prompt_tokens, completion_tokens, call_error, response
        except Exception as e:
            logging.info(f'Error occurred in ChatOllama call, retrying. Error type: {type(e).__name__}')
            time.sleep(5)
            retry_times += 1
            if retry_times == 10:
                logging.info('Retrying too many times')
                return None, None, True, None


def exec_action_click(info, web_ele, driver_task):
    # 在selenium執行點擊的動作
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    web_ele.click()
    time.sleep(3)


def exec_action_type(info, web_ele, driver_task):
    # 在selenium執行輸入的動作
    warn_obs = ""
    type_content = info['content']

    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
    try:
        web_ele.clear()
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except:
        pass

    actions = ActionChains(driver_task)
    actions.click(web_ele).perform()
    actions.pause(1)

    try:
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
    except:
        pass

    actions.send_keys(type_content)
    actions.pause(2)
    actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(10)
    return warn_obs


def exec_action_scroll(info, web_eles, driver_task, args, obs_info):
    # 在selenium執行滾動的動作
    scroll_ele_number = info['number']
    scroll_content = info['content']
    if scroll_ele_number == "WINDOW":
        if scroll_content == 'down':
            driver_task.execute_script(f"window.scrollBy(0, {args.window_height*2//3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, {-args.window_height*2//3});")
    else:
        if not args.text_only:
            scroll_ele_number = int(scroll_ele_number)
            web_ele = web_eles[scroll_ele_number]
        else:
            element_box = obs_info[scroll_ele_number]['union_bound']
            element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
            web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
    time.sleep(3)


def main():
    # 參數設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/test.json')
    parser.add_argument('--max_iter', type=int, default=5)
    # 原先的 api_key 參數可忽略或保留
    parser.add_argument("--api_key", default="key", type=str, help="YOUR_API_KEY")
    # api_model 參數改成對 ChatOllama 無用
    parser.add_argument("--api_model", default="llama3.1", type=str, help="model name")
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_attached_imgs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--text_only", action='store_true')
    # for web browser
    parser.add_argument("--headless", action='store_true', help='The window of selenium')
    parser.add_argument("--save_accessibility_tree", action='store_true')
    parser.add_argument("--force_device_scale", action='store_true')
    parser.add_argument("--window_width", type=int, default=1024)
    parser.add_argument("--window_height", type=int, default=768)
    parser.add_argument("--fix_box_color", action='store_true')

    args = parser.parse_args()

    # 使用 ChatOllama 初始化 llm，這裡依據 langchain_ollama 的使用方式設定參數
    llm = ChatOllama(
        model=args.api_model,
        temperature=args.temperature,
        # 可加入其他參數
    )

    options = driver_config(args)

    # Save Result file
    current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    result_dir = os.path.join(args.output_dir, current_time)
    os.makedirs(result_dir, exist_ok=True)

    # Load tasks
    tasks = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))

    for task_id in range(len(tasks)):  # 逐一處理每個task
        task = tasks[task_id]
        task_dir = os.path.join(result_dir, 'task{}'.format(task["id"]))
        os.makedirs(task_dir, exist_ok=True)
        setup_logger(task_dir)
        logging.info(f'########## TASK{task["id"]} ##########')

        driver_task = webdriver.Chrome(options=options)
        driver_task.set_window_size(args.window_width, args.window_height)
        driver_task.get(task['web'])
        try:
            driver_task.find_element(By.TAG_NAME, 'body').click()
        except:
            pass
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
        time.sleep(5)

        # 清空下載目錄
        for filename in os.listdir(args.download_dir):
            file_path = os.path.join(args.download_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        download_files = []

        fail_obs = ""
        pdf_obs = ""
        warn_obs = ""
        pattern = r'Thought:|Action:|Observation:'

        # 初始 prompt 設定
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "
        if args.text_only:
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT_TEXT_ONLY}]
            obs_prompt = "Observation: please analyze the accessibility tree and give the Thought and Action."

        init_msg = f"""Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. \n"""
        init_msg = init_msg.replace('https://www.example.com', task['web'])
        init_msg = init_msg + obs_prompt

        it = 0
        accumulate_prompt_token = 0
        accumulate_completion_token = 0

        while it < args.max_iter:
            logging.info(f'Iter: {it}')
            it += 1
            if not fail_obs:
                try:
                    if not args.text_only:
                        rects, web_eles, web_eles_text = get_web_element_rect(driver_task, fix_color=args.fix_box_color)
                    else:
                        accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                        ac_tree, obs_info = get_webarena_accessibility_tree(driver_task, accessibility_tree_path)
                except Exception as e:
                    if not args.text_only:
                        logging.error('Driver error when adding set-of-mark.')
                    else:
                        logging.error('Driver error when obtaining accessibility tree.')
                    logging.error(e)
                    break

                img_path = os.path.join(task_dir, 'screenshot{}.png'.format(it))
                driver_task.save_screenshot(img_path)

                if (not args.text_only) and args.save_accessibility_tree:
                    accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                    get_webarena_accessibility_tree(driver_task, accessibility_tree_path)

                b64_img = encode_image(img_path)

                if not args.text_only:
                    curr_msg = format_msg(it, init_msg, pdf_obs, warn_obs, b64_img, web_eles_text)
                else:
                    curr_msg = format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree)
                messages.append(curr_msg)
            else:
                curr_msg = {
                    'role': 'user',
                    'content': fail_obs
                }
                messages.append(curr_msg)

            if not args.text_only:
                messages = clip_message_and_obs(messages, args.max_attached_imgs)
            else:
                messages = clip_message_and_obs_text_only(messages, args.max_attached_imgs)

            # 呼叫 ChatOllama API
            prompt_tokens, completion_tokens, call_error, llm_response = call_chatollama_api(args, llm, messages)

            if call_error:
                break
            else:
                accumulate_prompt_token += prompt_tokens
                accumulate_completion_token += completion_tokens
                logging.info(f'Accumulate Prompt Tokens: {accumulate_prompt_token}; Accumulate Completion Tokens: {accumulate_completion_token}')
                logging.info('API call complete...')
            # 將回應加入對話紀錄
            messages.append({'role': 'assistant', 'content': llm_response})

            if (not args.text_only) and rects:
                logging.info(f"Num of interactive elements: {len(rects)}")
                for rect_ele in rects:
                    driver_task.execute_script("arguments[0].remove()", rect_ele)
                rects = []

            try:
                assert 'Thought:' in llm_response and 'Action:' in llm_response
            except AssertionError as e:
                logging.error(e)
                fail_obs = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
                continue

            chosen_action = re.split(pattern, llm_response)[2].strip()
            action_key, info = extract_information(chosen_action)

            fail_obs = ""
            pdf_obs = ""
            warn_obs = ""
            try:
                window_handle_task = driver_task.current_window_handle
                driver_task.switch_to.window(window_handle_task)

                if action_key == 'click':
                    if not args.text_only:
                        click_ele_number = int(info[0])
                        web_ele = web_eles[click_ele_number]
                    else:
                        click_ele_number = info[0]
                        element_box = obs_info[click_ele_number]['union_bound']
                        element_box_center = (element_box[0] + element_box[2] // 2,
                                              element_box[1] + element_box[3] // 2)
                        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
                    exec_action_click(info, web_ele, driver_task)
                    current_files = sorted(os.listdir(args.download_dir))
                    if current_files != download_files:
                        time.sleep(10)
                        current_files = sorted(os.listdir(args.download_dir))
                        current_download_file = [pdf_file for pdf_file in current_files if pdf_file not in download_files and pdf_file.endswith('.pdf')]
                        if current_download_file:
                            pdf_file = current_download_file[0]
                            pdf_obs = get_pdf_retrieval_ans_from_assistant(llm, os.path.join(args.download_dir, pdf_file), task['ques'])
                            shutil.copy(os.path.join(args.download_dir, pdf_file), task_dir)
                            pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                        download_files = current_files
                    if web_ele.tag_name.lower() == 'button' and web_ele.get_attribute("type") == 'submit':
                        time.sleep(10)

                elif action_key == 'wait':
                    time.sleep(5)

                elif action_key == 'type':
                    if not args.text_only:
                        type_ele_number = int(info['number'])
                        web_ele = web_eles[type_ele_number]
                    else:
                        type_ele_number = info['number']
                        element_box = obs_info[type_ele_number]['union_bound']
                        element_box_center = (element_box[0] + element_box[2] // 2,
                                              element_box[1] + element_box[3] // 2)
                        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
                    warn_obs = exec_action_type(info, web_ele, driver_task)
                    if 'wolfram' in task['web']:
                        time.sleep(5)

                elif action_key == 'scroll':
                    if not args.text_only:
                        exec_action_scroll(info, web_eles, driver_task, args, None)
                    else:
                        exec_action_scroll(info, None, driver_task, args, obs_info)

                elif action_key == 'goback':
                    driver_task.back()
                    time.sleep(2)

                elif action_key == 'google':
                    driver_task.get('https://www.google.com/')
                    time.sleep(2)

                elif action_key == 'answer':
                    logging.info(info['content'])
                    logging.info('finish!!')
                    break

                else:
                    raise NotImplementedError
                fail_obs = ""
            except Exception as e:
                logging.error('driver error info:')
                logging.error(e)
                if 'element click intercepted' not in str(e):
                    fail_obs = "The action you have chosen cannot be exected. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
                else:
                    fail_obs = ""
                time.sleep(2)

        print_message(messages, task_dir)
        driver_task.quit()
        logging.info(f'Total cost: {accumulate_prompt_token / 1000 * 0.01 + accumulate_completion_token / 1000 * 0.03}')

if __name__ == '__main__':
    main()
    print('End of process')
