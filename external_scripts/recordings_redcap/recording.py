import uuid
import pandas as pd
import argparse
import re
acoustic_names_mapping = { # skim over the acoustic tasks.....
    "picturesAndDoors": "Conversation (2 to 4)-book",
    "abcs": "Abcs and 123s",
    "123s": "Abcs and 123s",
    "longSounds": "Long Sounds",
    "noisySounds": "Noisy Sounds",
    "sillySounds": "Silly Sounds",
    "pictureDescription": "Picture Description",
    "picture": "Identifying Pictures",
    "days": "Days and Number naming ",
    "months": "Days and Number naming ",
    "sentence": "Repeating Sentences",
    "conversation": "Conversation", 
    "roleNaming": "Role naming tasks sounds",
    "passage": "Reading Passages",
    "namingAnimals": "Generative Naming Task",
    "namingFood": "Generative Naming Task",
    "repeatWords": "Repeating Words"

}


recording_names_mapping = {
    "picturesAndDoors": "Conversation (2 to 4)-book",
    "abcs": "Abcs and 123s-abcs",
    "123s": "Abcs and 123s-123s",
    "noisySounds1": "Noisy Sounds-1",
    "noisySounds2": "Noisy Sounds-2",
    "noisySounds3": "Noisy Sounds-3",
    "noisySounds4": "Noisy Sounds-4",
    "noisySounds5": "Noisy Sounds-5",
    "noisySounds6": "Noisy Sounds-6",
    "noisySounds7": "Noisy Sounds-7",
    "noisySounds8": "Noisy Sounds-8",
    "noisySounds9": "Noisy Sounds-9",
    "noisySounds10": "Noisy Sounds-10",
    "picture1": "Identifying Pictures-1",
    "picture2": "Identifying Pictures-2",
    "picture3": "Identifying Pictures-3",
    "picture4": "Identifying Pictures-4",
    "picture5": "Identifying Pictures-5",
    "picture6": "Identifying Pictures-6",
    "picture7": "Identifying Pictures-7",
    "picture8": "Identifying Pictures-8",
    "picture9": "Identifying Pictures-9",
    "picture10": "Identifying Pictures-10",
    "picture11": "Identifying Pictures-11",
    "picture12": "Identifying Pictures-12",
    "picture13": "Identifying Pictures-13",
    "picture14": "Identifying Pictures-14",
    "picture15": "Identifying Pictures-15",
    "picture16": "Identifying Pictures-16",
    "picture17": "Identifying Pictures-17",
    "picture18": "Identifying Pictures-18",
    "picture19": "Identifying Pictures-19",
    "picture20": "Identifying Pictures-20",
    "picture21": "Identifying Pictures-21",
    "picture22": "Identifying Pictures-22",
    "picture23": "Identifying Pictures-23",
    "picture24": "Identifying Pictures-24",
    "picture25": "Identifying Pictures-25",
    "picture26": "Identifying Pictures-26",
    "picture27": "Identifying Pictures-27",
    "picture28": "Identifying Pictures-28",
    "picture29": "Identifying Pictures-29",
    "picture30": "Identifying Pictures-30",
    "picture31": "Identifying Pictures-31",
    "picture32": "Identifying Pictures-32",
    "picture33": "Identifying Pictures-33",
    "picture34": "Identifying Pictures-34",
    "picture35": "Identifying Pictures-35",
    "picture36": "Identifying Pictures-36",
    "picture37": "Identifying Pictures-37",
    "days": "Days and Number naming-days",
    "months": "Days and Number naming-months",
    "sillySounds1": "Silly Sounds-1",
    "sillySounds2": "Silly Sounds-2",
    "sillySounds3": "Silly Sounds-3",
    "sillySounds4": "Silly Sounds-4",
    "sentence1": "Repeating Sentences-1",
    "sentence2": "Repeating Sentences-2",
    "sentence3": "Repeating Sentences-3",
    "sentence4": "Repeating Sentences-4",
    "sentence5": "Repeating Sentences-5",
    "sentence6": "Repeating Sentences-6",
    "sentence7": "Repeating Sentences-7",
    "pictureDescription1": "Picture Description",
    "conversation(10+)1": "Conversation (10 plus)-favorite_food",
    "conversation(10+)2": "Conversation (10 plus)-favorite_show_movie_game",
    "conversation(10+)3": "Conversation (10 plus)-outside_of_school",
    "conversation(10+)4": "Conversation (10 plus)-ready_for_school",
    "conversation(6to10)1": "Conversation (6 plus)-favorite_food",
    "conversation(6to10)2": "Conversation (6 plus)-favorite_show_movie_game",
    "conversation(6to10)3": "Conversation (6 plus)-outside_of_school",
    "conversation(6to10)4": "Conversation (6 plus)-ready_for_school",
    "conversation(4to6)1": "Conversation (4 to 6)-book",
    "conversation(2to4)1": "Conversation (2 to 4)-book",    
    # "conversation4": "Conversation (6 plus)-ready_for_school",
    # "conversation2": "Conversation (6 plus)-favorite_show_movie_game",
    # "conversation1": "Conversation (6 plus)-favorite_food",
    # "conversation3": "Conversation (6 plus)-outside_of_school",
    "roleNaming1": "Role naming tasks sounds-days",
    "roleNaming2": "Role naming tasks sounds-months",
    "roleNaming3": "Role naming tasks sounds-numbers",
    "passage1": "Reading Passage-1",
    "passage2": "Reading Passage-2",
    "passage3": "Reading Passage-3",
    "passage4": "Reading Passage-4",
    "passage5": "Reading Passage-5",
    "passage6": "Reading Passage-6",
    "passage7": "Reading Passage-7",
    "passage8": "Reading Passage-8",
    "passage9": "Reading Passage-9",
    "passage10": "Reading Passage-10",
    "passage11": "Reading Passage-11",
    "generativeNamingTask1": "Generative Naming Task-animals",
    "generativeNamingTask2": "Generative Naming Task-food",
    "longSounds1": "Long Sounds-1",
    "longSounds2": "Long Sounds-2",
    "repeatWords1": "Repeating Words-smile",
    "repeatWords2": "Repeating Words-great",
    "repeatWords3": "Repeating Words-sled",
    "repeatWords4": "Repeating Words-slip",
    "repeatWords5": "Repeating Words-pants",
    "repeatWords6": "Repeating Words-bad",
    "repeatWords7": "Repeating Words-pinch",
    "repeatWords8": "Repeating Words-such",
    "repeatWords9": "Repeating Words-take",
    "repeatWords10": "Repeating Words-need",
    "repeatWords11": "Repeating Words-scab",
    "repeatWords12": "Repeating Words-five",
    "repeatWords13": "Repeating Words-class",
    "repeatWords14": "Repeating Words-mouth",
    "repeatWords15": "Repeating Words-me",
    "repeatWords16": "Repeating Words-fed",
    "repeatWords17": "Repeating Words-beef",
    "repeatWords18": "Repeating Words-fold",
    "repeatWords19": "Repeating Words-hunt",
    "repeatWords20": "Repeating Words-no",
    "repeatWords21": "Repeating Words-are",
    "repeatWords22": "Repeating Words-pond",
    "repeatWords23": "Repeating Words-teach",
    "repeatWords24": "Repeating Words-slice",
    "repeatWords25": "Repeating Words-tree"
}



def create_recording_redcap(recording_csv, output_csv):

    df = pd.read_csv(recording_csv, dtype="str")
    # Clean column names first
    df.columns = df.columns.str.strip().str.replace('\u2060', '', regex=False)
    l = []
    for col in df.columns:
        if df[col].isnull().all():
            l.append(col)

    l += [
        "peds_mc_neck_mass___thyroglossal_duct_cyst",
        "peds_mc_neck_mass___branchial_cleft_cyst",
        "peds_mc_neck_mass___dermoid_cyst",
        "peds_mc_neck_mass___enlarged_lymph_node",
        "participant_study_id",
        # "recording_profile_name",
        # "recording_microphone",
        
    ]
    df.drop(columns=l, axis=1, inplace=True)

    # Convert recording_id from 32 char hex to standard UUID format
    # df["recording_id"] = df["recording_id"].apply(
    #     lambda x: str(uuid.UUID(x)) if pd.notna(x) else x
    # )
    
    df["recording_id"] = df["recording_id"].apply(
    lambda x: re.sub(r"([a-f0-9]{32})", lambda m: f"-{str(uuid.UUID(m.group()))}", x, flags=re.IGNORECASE) if pd.notna(x) else x
)   
    
    uuid_cols = ["acoustic_task_session_id", "session_id","acoustic_task_id", "recording_acoustic_task_id", "recording_session_id"]
    for col in uuid_cols:
        df[col] = df[col].apply(lambda x: str(uuid.UUID(x)) if pd.notna(x) else x)
    
    # df["acoustic_task_session_id"] = df["acoustic_task_session_id"].apply(
    #     lambda x: str(uuid.UUID(x)) if pd.notna(x) else x
    # )
    
    df["acoustic_task_name"] = df["acoustic_task_name"].apply(
        lambda x: acoustic_names_mapping[x] if pd.notna(x) and x in acoustic_names_mapping else x
    )
    
    df["recording_name"] = df["recording_name"].apply(
        lambda x: recording_names_mapping[x] if pd.notna(x) and x in recording_names_mapping else x
    )
    
    df["recording_input_gain"] =  ""
    df["recording_file_share"]=   ""
    df["recording_storage_account"] =  ""

    df.to_csv(f"{output_csv}/recording-final-jun8.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_csv", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    
    create_recording_redcap(args.recording_csv, args.output_path)
    