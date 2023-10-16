import uuid

DEFAULT_LANGUAGE_MODEL = 'google/flan-t5-xl'
DEFAULT_TASK_DESCRIPTION = 'prompts/task_description_5.txt'
DEFAULT_NUM_SHOTS = 2
DEFAULT_JACCARD_SAMPLING = 'TRUE'
DEFAULT_USE_BULLET_POINTED_GOAL = 'FALSE'
DEFAULT_TEMPERATURE = 0.8
DEFAULT_SYS_NLU = 'TRUE'

DEFAULT_PRINT_DETAILS = 'FALSE'
DEFAULT_TOTAL_DIALOGS = 200

# README: need to be adapted if needed
exp_name = uuid.uuid4().hex
DEFAULT_SAVE_DIR = 'results/{}'.format(exp_name)

PROMPT_ENGINEERING_ONLY = False

language_model_list = [
    DEFAULT_LANGUAGE_MODEL,
    'decapoda-research/llama-7b-hf',
    'nomic-ai/gpt4all-j',
    'azure_openai/gpt-35-turbo'
]
task_description_list = [
    DEFAULT_TASK_DESCRIPTION,
    'prompts/task_description_8.txt',  # like 5 + domain names
    'prompts/task_description_6.txt',  # lenghty description
    'prompts/task_description_9.txt',  # like 5, but only 2 sencences
    'prompts/task_description_7.txt',  # empty task description
]
num_shots_list = [
    DEFAULT_NUM_SHOTS,
    1,
    0,
    3,
    4
]
jaccard_sampling_list = [
    DEFAULT_JACCARD_SAMPLING,
    'FALSE'
]
use_bullet_pointed_goal_list = [
    DEFAULT_USE_BULLET_POINTED_GOAL,
    'TRUE'
]
sys_nlu_list = [
    DEFAULT_SYS_NLU,
    'FALSE'
]
temperature_list = [
    DEFAULT_TEMPERATURE,
    0.7,
    0.9
]

output_filename = 'scripts/run_experiment.sh'
fp_out = open(output_filename, 'w')

SCRIPT_NAME = 'python tutorials/user_simulator_script_ours.py '

COMMAND = (
        '--model-id {} --print-details {} --total-dialog {} '
        '--temperature {} --num-shots {} --use-bullet-pointed-goal {} '
        '--jaccard-sampling {} --task-description-file-path {} '
        '--sys-nlu {}'
)

commands = set()


def get_save_dir_name(cmd):
    return '/{}'.format(
        cmd.replace(' --', '\\|')
           .replace('--', '\\|')
           .replace(' ', '\\:')
           .replace('/', '_')
    )


def write_line(text, fp_out):
    fp_out.write(f"{text}\n")


write_line(text="# ************************************", fp_out=fp_out)
write_line(text="# VARY ONE DIM, KEEP OTHER DIM DEFAULT", fp_out=fp_out)
write_line(text="# ************************************", fp_out=fp_out)

write_line(text="# language_model", fp_out=fp_out)
for language_model in language_model_list:
    command = (
        COMMAND.format(
            language_model,
            DEFAULT_PRINT_DETAILS,
            DEFAULT_TOTAL_DIALOGS,
            DEFAULT_TEMPERATURE,
            DEFAULT_NUM_SHOTS,
            DEFAULT_USE_BULLET_POINTED_GOAL,
            DEFAULT_JACCARD_SAMPLING,
            DEFAULT_TASK_DESCRIPTION,
            DEFAULT_SYS_NLU
        )
    )
    save_dir = DEFAULT_SAVE_DIR + get_save_dir_name(command)
    command = SCRIPT_NAME + command + ' --save-dir {}'.format(save_dir)
    write_line(text=command, fp_out=fp_out)

write_line(text="# num_shots", fp_out=fp_out)
for num_shots in num_shots_list:
    command = (
        COMMAND.format(
            DEFAULT_LANGUAGE_MODEL,
            DEFAULT_PRINT_DETAILS,
            DEFAULT_TOTAL_DIALOGS,
            DEFAULT_TEMPERATURE,
            num_shots,
            DEFAULT_USE_BULLET_POINTED_GOAL,
            DEFAULT_JACCARD_SAMPLING,
            DEFAULT_TASK_DESCRIPTION,
            DEFAULT_SYS_NLU
        )
    )
    save_dir = DEFAULT_SAVE_DIR + get_save_dir_name(command)
    command = SCRIPT_NAME + command + ' --save-dir {}'.format(save_dir)
    write_line(text=command, fp_out=fp_out)

write_line(text="# task_description", fp_out=fp_out)
for task_description in task_description_list:
    command = (
        COMMAND.format(
            DEFAULT_LANGUAGE_MODEL,
            DEFAULT_PRINT_DETAILS,
            DEFAULT_TOTAL_DIALOGS,
            DEFAULT_TEMPERATURE,
            DEFAULT_NUM_SHOTS,
            DEFAULT_USE_BULLET_POINTED_GOAL,
            DEFAULT_JACCARD_SAMPLING,
            task_description,
            DEFAULT_SYS_NLU
        )
    )
    save_dir = DEFAULT_SAVE_DIR + get_save_dir_name(command)
    command = SCRIPT_NAME + command + ' --save-dir {}'.format(save_dir)
    write_line(text=command, fp_out=fp_out)

write_line(text="# use_bullet_pointed_goal", fp_out=fp_out)
for use_bullet_pointed_goal in use_bullet_pointed_goal_list:
    command = (
        COMMAND.format(
            DEFAULT_LANGUAGE_MODEL,
            DEFAULT_PRINT_DETAILS,
            DEFAULT_TOTAL_DIALOGS,
            DEFAULT_TEMPERATURE,
            DEFAULT_NUM_SHOTS,
            use_bullet_pointed_goal,
            DEFAULT_JACCARD_SAMPLING,
            DEFAULT_TASK_DESCRIPTION,
            DEFAULT_SYS_NLU
        )
    )
    save_dir = DEFAULT_SAVE_DIR + get_save_dir_name(command)
    command = SCRIPT_NAME + command + ' --save-dir {}'.format(save_dir)
    write_line(text=command, fp_out=fp_out)

write_line(text="# jaccard_sampling", fp_out=fp_out)
for jaccard_sampling in jaccard_sampling_list:
    command = (
        COMMAND.format(
            DEFAULT_LANGUAGE_MODEL,
            DEFAULT_PRINT_DETAILS,
            DEFAULT_TOTAL_DIALOGS,
            DEFAULT_TEMPERATURE,
            DEFAULT_NUM_SHOTS,
            DEFAULT_USE_BULLET_POINTED_GOAL,
            jaccard_sampling,
            DEFAULT_TASK_DESCRIPTION,
            DEFAULT_SYS_NLU
        )
    )
    save_dir = DEFAULT_SAVE_DIR + get_save_dir_name(command)
    command = SCRIPT_NAME + command + ' --save-dir {}'.format(save_dir)
    write_line(text=command, fp_out=fp_out)

write_line(text="# temperature", fp_out=fp_out)
for temperature in temperature_list:
    command = (
        COMMAND.format(
            DEFAULT_LANGUAGE_MODEL,
            DEFAULT_PRINT_DETAILS,
            DEFAULT_TOTAL_DIALOGS,
            temperature,
            DEFAULT_NUM_SHOTS,
            DEFAULT_USE_BULLET_POINTED_GOAL,
            DEFAULT_JACCARD_SAMPLING,
            DEFAULT_TASK_DESCRIPTION,
            DEFAULT_SYS_NLU
        )
    )
    save_dir = DEFAULT_SAVE_DIR + get_save_dir_name(command)
    command = SCRIPT_NAME + command + ' --save-dir {}'.format(save_dir)
    write_line(text=command, fp_out=fp_out)

write_line(text="# sys_nlu", fp_out=fp_out)
for sys_nlu in sys_nlu_list:
    command = (
        COMMAND.format(
            DEFAULT_LANGUAGE_MODEL,
            DEFAULT_PRINT_DETAILS,
            DEFAULT_TOTAL_DIALOGS,
            DEFAULT_TEMPERATURE,
            DEFAULT_NUM_SHOTS,
            DEFAULT_USE_BULLET_POINTED_GOAL,
            DEFAULT_JACCARD_SAMPLING,
            DEFAULT_TASK_DESCRIPTION,
            sys_nlu
        )
    )
    save_dir = DEFAULT_SAVE_DIR + get_save_dir_name(command)
    command = SCRIPT_NAME + command + ' --save-dir {}'.format(save_dir)
    write_line(text=command, fp_out=fp_out)

write_line(text="# ************************************", fp_out=fp_out)
write_line(text="# NGHIA PROMPT EXPERIMENTS ONLY", fp_out=fp_out)
write_line(text="# ************************************", fp_out=fp_out)
for task_description in task_description_list:
    for num_shots in num_shots_list:
        for use_bullet_pointed_goal in use_bullet_pointed_goal_list:

            command = (
                COMMAND.format(
                    DEFAULT_LANGUAGE_MODEL,
                    DEFAULT_PRINT_DETAILS,
                    DEFAULT_TOTAL_DIALOGS,
                    DEFAULT_TEMPERATURE,
                    num_shots,
                    use_bullet_pointed_goal,
                    DEFAULT_JACCARD_SAMPLING,
                    task_description,
                    DEFAULT_SYS_NLU
                )
            )
            save_dir = DEFAULT_SAVE_DIR + get_save_dir_name(command)
            command = SCRIPT_NAME + command + ' --save-dir {}'.format(save_dir)
            write_line(text=command, fp_out=fp_out)

write_line(text="# ************************************", fp_out=fp_out)
write_line(text="# MF PROMPT EXPERIMENTS, April 26", fp_out=fp_out)
write_line(text="# ************************************", fp_out=fp_out)
for task_description in task_description_list:
    # extra experiments if default task description:
    #  more shots: 0 to 6
    #  bullet poinst on and off
    if task_description == DEFAULT_TASK_DESCRIPTION:
        for num_shots in list(range(0, 7)):
            for use_bullet_pointed_goal in use_bullet_pointed_goal_list:
                # if 0-shot, there will be no requirements anyways
                if num_shots == 0 and use_bullet_pointed_goal == 'TRUE':
                    continue
                # if 4+ shots, don't care about adding bullets or not
                if num_shots > 3 and use_bullet_pointed_goal == 'TRUE':
                    continue
                command = (
                    COMMAND.format(
                        DEFAULT_LANGUAGE_MODEL,
                        DEFAULT_PRINT_DETAILS,
                        DEFAULT_TOTAL_DIALOGS,
                        DEFAULT_TEMPERATURE,
                        num_shots,
                        use_bullet_pointed_goal,
                        DEFAULT_JACCARD_SAMPLING,
                        task_description,
                        DEFAULT_SYS_NLU
                    )
                )
                save_dir = DEFAULT_SAVE_DIR + get_save_dir_name(command)
                command = SCRIPT_NAME + command + ' --save-dir {}'.format(save_dir)
                write_line(text=command, fp_out=fp_out)
    else:
        for num_shots in num_shots_list:
            # not interested in 4+ shots...
            if num_shots > 3:
                continue
            command = (
                COMMAND.format(
                    DEFAULT_LANGUAGE_MODEL,
                    DEFAULT_PRINT_DETAILS,
                    DEFAULT_TOTAL_DIALOGS,
                    DEFAULT_TEMPERATURE,
                    num_shots,
                    DEFAULT_USE_BULLET_POINTED_GOAL,
                    DEFAULT_JACCARD_SAMPLING,
                    task_description,
                    DEFAULT_SYS_NLU
                )
            )
            save_dir = DEFAULT_SAVE_DIR + get_save_dir_name(command)
            command = SCRIPT_NAME + command + ' --save-dir {}'.format(save_dir)
            write_line(text=command, fp_out=fp_out)


write_line(text="# ************************************", fp_out=fp_out)
write_line(text="# ROLAND GPT35 PROMPT EXPERIMENTS", fp_out=fp_out)
write_line(text="# ************************************", fp_out=fp_out)
for language_model in ['azure_openai/gpt-35-turbo']:
    for num_shots in num_shots_list:
        for task_description in task_description_list:
            command = (
                COMMAND.format(
                    language_model,
                    DEFAULT_PRINT_DETAILS,
                    DEFAULT_TOTAL_DIALOGS,
                    DEFAULT_TEMPERATURE,
                    num_shots,
                    DEFAULT_USE_BULLET_POINTED_GOAL,
                    DEFAULT_JACCARD_SAMPLING,
                    task_description,
                    DEFAULT_SYS_NLU
                )
            )
            save_dir = DEFAULT_SAVE_DIR + get_save_dir_name(command)
            command = SCRIPT_NAME + command + ' --save-dir {}'.format(
                save_dir)
            write_line(text=command, fp_out=fp_out)

# write file
fp_out.close()
