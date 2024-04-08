import datetime
import platform
import re
import shlex
import shutil
import tempfile
import time
from pathlib import Path

bl_info = {
    "name": "Voice-Based 3D Modeling",
    "blender": (2, 82, 0),
    "category": "Object",
    "author": "Andrea Esposito (andrea.esposito@uniba.it)",
    "version": (0, 0, 1),
    "location": "3D View > UI > Voice Based Modeling",
    "description": "A voice assistant for Blender",
    "warning": "Experimental",
}

import ensurepip
import os
import subprocess
import sys

import bpy

REQUIREMENTS = ["openai", "sounddevice", "scipy", "soundfile"]

ensurepip.bootstrap()
os.environ.pop("PIP_REQ_TRACKER", None)
subprocess.check_output([sys.executable, '-m', 'pip', 'install', *REQUIREMENTS])

system_prompt = """You are an assistant made for the purposes of helping the user with Blender, the 3D software. 
- Respond with your answers in markdown (```). 
- Preferably import entire modules instead of bits. 
- Do not perform destructive operations on the meshes. 
- Do not use cap_ends. Do not do more than what is asked (setting up render settings, adding cameras, etc)
- Do not respond with anything that is not Python code.

Example:

user: create 10 cubes in random locations from -10 to 10
assistant:
```
import bpy
import random
bpy.ops.mesh.primitive_cube_add()

#how many cubes you want to add
count = 10

for c in range(0,count):
    x = random.randint(-10,10)
    y = random.randint(-10,10)
    z = random.randint(-10,10)
    bpy.ops.mesh.primitive_cube_add(location=(x,y,z))
```"""


class VoiceAssistantPanel(bpy.types.Panel):
    """Object Cursor Array"""
    bl_idname = "OBJECT_PT_voice3d.panel"
    bl_label = "Voice 3D Assistant"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = bl_label
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        column = layout.column(align=True)
        addon_prefs = context.preferences.addons[__name__].preferences
        column.prop(addon_prefs, "audio_input_device")
        column.separator()
        row = column.row(align=True)
        action = StartRecording if not context.scene.voice3d_is_recording else StopRecording
        row.operator(StartRecording.bl_idname, text=action.bl_label)


import logging

logger = logging.getLogger(__name__)

ffmpeg_exe_path = shutil.which("ffmpeg")
from string import whitespace


# Voice recording implementation from:
# https://github.com/britalmeida/push_to_talk
def get_audio_devices_list_linux():
    """Get list of audio devices on Linux."""

    # Get named devices using ALSA and arecord.
    arecord_exe_path = shutil.which("arecord")
    if not arecord_exe_path:
        return []

    sound_cards = []
    with subprocess.Popen(args=[arecord_exe_path, "-L"], stdout=subprocess.PIPE) as proc:
        arecord_output = proc.stdout.read()
        for line in arecord_output.splitlines():
            line = line.decode('utf-8')

            # Skip indented lines, search only for PCM names
            if line.startswith(tuple(w for w in whitespace)) == False:
                # Show only names which are likely to be an input device.
                # Skip names that are known to be something else.
                if not (line in ["null", "oss", "pulse", "speex"] or
                        line.startswith(("surround", "usbstream", "front")) or
                        line.endswith(("rate", "mix", "snoop"))):
                    # Found one!
                    sound_cards.append(line)

    return sound_cards


def get_audio_devices_list_darwin():
    """Get list of audio devices on macOS."""

    if not ffmpeg_exe_path:
        return []
    args = [ffmpeg_exe_path] + shlex.split("-f avfoundation -list_devices true -hide_banner -i dummy")

    av_device_lines = []
    with subprocess.Popen(args=args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        command_output = proc.stderr.read()
        for line in command_output.splitlines():
            line = line.decode('utf-8')

            if line.startswith("[AVFoundation"):
                av_device_lines.append(line)

    sound_cards = []

    # Strip video devices from list
    include_entries = False
    for av_device_line in av_device_lines:
        if 'AVFoundation video devices:' in av_device_line:
            include_entries = False
        elif 'AVFoundation audio devices:' in av_device_line:
            include_entries = True
        # When in the "audio devices" part of the list, include entries.
        elif include_entries:
            sound_cards.append(av_device_line)

    # Parse the remaining items so they go from:
    # [AVFoundation input device @ 0x7f9c0a604340] [0] Unknown USB Audio Device
    # to:
    # Unknown USB Audio Device"
    pattern = r'\[.*?\]'
    sound_cards = [re.sub(pattern, '', sound_card) for sound_card in sound_cards]
    # Important: we assume that the device number (e.g. [0]) matches the order
    # of the device in the list. This is used to build the ffmpeg command in
    # the start_recording function.
    return sound_cards


def get_audio_devices_list_windows():
    """Get list of audio devices on Windows."""

    if not ffmpeg_exe_path:
        return []
    args = [ffmpeg_exe_path] + shlex.split("-f dshow -list_devices true -hide_banner -i dummy")

    # dshow list_devices may output either individual devices tagged with '(audio)', e.g.:
    # [dshow @ 00000137146e4800] "Microphone (HD Pro Webcam)" (audio)
    # or all audio devices grouped after a 'DirectShow audio devices' heading, e.g.:
    # [dshow @ 02cec400] DirectShow audio devices
    # [dshow @ 02cec400]  "Desktop Microphone (3- Studio -"
    # [dshow @ 02cec400]     Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\Desktop Microphone (3- Studio -"
    grouped_output_version = False

    av_device_lines = []
    with subprocess.Popen(args=args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        command_output = proc.stderr.read()
        for line in command_output.splitlines():
            line = line.decode('utf-8')

            # Start by optimistically adding all (audio) lines, regardless of mode.
            # What could possibly go wrong? right?
            if line.endswith("(audio)"):
                av_device_lines.append(line)
            # If at some point we find the audio header, we switch mode.
            elif "DirectShow audio devices" in line:
                grouped_output_version = True
            # In grouped mode, add lines that should have device names (skip input file errors).
            elif grouped_output_version:
                if "Alternative name" not in line and "Error" not in line:
                    av_device_lines.append(line)

    # Extract the device names from the lines.
    sound_cards = []
    pattern = r'\[.*?\]'
    for av_device_line in av_device_lines:
        names_within_quotes = re.findall(r'"(.+?)"', av_device_line)
        if len(names_within_quotes) == 1:
            sound_cards.append(names_within_quotes[0])
        else:
            # Keep it for the user to see, it might help them figure out their audio setup.
            sound_cards.append(f"error parsing entry '{av_device_line}'")

    return sound_cards


os_platform = platform.system()
supported_platforms = {"Windows", "Linux", "Darwin"}


def get_audio_devices():
    logger.debug("Polling system sound cards to update audio input drop-down")

    # Detect existing sound cards and devices
    if os_platform == 'Linux':
        sound_cards = get_audio_devices_list_linux()
    elif os_platform == 'Darwin':
        sound_cards = get_audio_devices_list_darwin()
    elif os_platform == 'Windows':
        sound_cards = get_audio_devices_list_windows()

    if not sound_cards:
        sound_cards = ["no audio device found"]

    return sound_cards


def populate_enum_items_for_sound_devices(self, context):
    """Query the system for available audio devices and populate enum items."""

    # Re-use the existing enum values if they weren't generated too long ago.
    # Note: this generate function is called often, on draw of the UI element
    # that renders the enum property and per each enum item when the dropdown
    # is expanded.
    # To avoid bogging down the UI render pass, we avoid calling this function
    # too often, but we still want to call it occasionally, in case the user
    # plugs in a new audio device while Blender is running.
    try:
        last_executed = populate_enum_items_for_sound_devices.last_executed
        if (time.time() - last_executed) < 5:  # seconds
            return populate_enum_items_for_sound_devices.enum_items
    except AttributeError:
        # First time that the enum is being generated.
        pass

    logger.debug("Polling system sound cards to update audio input drop-down")

    # Detect existing sound cards and devices
    if os_platform == 'Linux':
        sound_cards = get_audio_devices_list_linux()
    elif os_platform == 'Darwin':
        sound_cards = get_audio_devices_list_darwin()
    elif os_platform == 'Windows':
        sound_cards = get_audio_devices_list_windows()

    if not sound_cards:
        sound_cards = ["no audio device found"]

    # Generate items to show in the enum dropdown.
    # TODO: get_audio_devices functions could return the full tuple instead, e.g.:
    # linux: [("sysdefault:CARD=PCH", "HDA Intel PCH, ALC269VC Analog", "Default Audio Device")]
    # macOS: [(0, "Unknown USB Audio Device", "Unknown USB Audio Device")]
    enum_items = []
    for idx, sound_card in enumerate(sound_cards):
        enum_value = f"{idx}" if os_platform == 'Darwin' else sound_card
        enum_items.append((enum_value, sound_card, sound_card))

    # Update the cached enum items and the generation timestamp
    populate_enum_items_for_sound_devices.enum_items = enum_items
    populate_enum_items_for_sound_devices.last_executed = time.time()

    logger.debug(f"Scanned & found sound devices: {populate_enum_items_for_sound_devices.enum_items}")
    return populate_enum_items_for_sound_devices.enum_items


def save_sound_card_preference(self, context):
    """Sync the chosen audio device to the user preferences.

    Called when the enum property is set.
    Sync the chosen audio device from the UI enum to the persisted user
    preferences according to the current OS.
    """

    addon_prefs = context.preferences.addons[__name__].preferences
    audio_device = addon_prefs.audio_input_device

    logger.debug(f"Set audio input preference to '{audio_device}' for {os_platform}")

    if os_platform == 'Linux':
        addon_prefs.audio_device_linux = audio_device
    elif os_platform == 'Darwin':
        addon_prefs.audio_device_darwin = audio_device
    elif os_platform == 'Windows':
        addon_prefs.audio_device_windows = audio_device


from bpy.props import StringProperty, EnumProperty


class SEQUENCER_PushToTalk_Preferences(bpy.types.AddonPreferences):
    bl_idname = "voicebased3dmodelling"

    prefix: StringProperty(
        name="Prefix",
        description="A label to name the created sound strips and files",
        default="temp_dialog",
    )
    sounds_dir: StringProperty(
        name="Sounds",
        description="Directory where to save the generated audio files",
        default="//",
        subtype="FILE_PATH",
    )
    # Explicitly save an audio configuration per platform in case the same user uses Blender in
    # different platforms and syncs user settings.
    audio_device_linux: StringProperty(
        name="Audio Input Device (Linux)",
        description="If automatic detection of the sound card fails, "
                    "manually insert a value given by 'arecord -L'",
        default="default",
    )
    audio_device_darwin: StringProperty(
        name="Audio Input Device (macOS)",
        description="Hardware slot of the audio input device given by 'ffmpeg'",
        default="setting not synced yet",
    )
    audio_device_windows: StringProperty(
        name="Audio Input Device (Windows)",
        description="Hardware slot of the audio input device given by 'ffmpeg'",
        default="setting not synced yet",
    )
    # The runtime audio device, depending on platform.
    audio_input_device: EnumProperty(
        items=populate_enum_items_for_sound_devices,
        name="Sound Card",
        description="Sound card to be used, from the ones found on this computer",
        options={'SKIP_SAVE'},
        update=save_sound_card_preference,
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        problem_found = ""
        if os_platform not in supported_platforms:
            problem_found = f"Recording on {os_platform} is not supported"
        elif not ffmpeg_exe_path:
            problem_found = "ffmpeg not found separately installed"

        col = layout.column()
        if problem_found:
            col.label(text=problem_found, icon='ERROR')
            col = col.column()
            col.enabled = False

        addon_prefs = context.preferences.addons[__name__].preferences

        col.prop(addon_prefs, "prefix")
        col.prop(addon_prefs, "sounds_dir")

        col.separator()
        col.prop(addon_prefs, "audio_input_device")
        # DEBUG
        # col.prop(addon_prefs, "audio_device_linux", text="(linux Debug)")
        # col.prop(addon_prefs, "audio_device_darwin", text="(macOS Debug)")
        # col.prop(addon_prefs, "audio_device_windows", text="(Win Debug)")

        # Show a save button for the user preferences if they aren't automatically saved.
        prefs = context.preferences
        if not prefs.use_preferences_save:
            col.separator()
            col.operator(
                "wm.save_userpref",
                text=f"Save Preferences{' *' if prefs.is_dirty else ''}",
            )


filepath = tempfile.NamedTemporaryFile(suffix=".wav", delete=True)
filepath.close()


class StopRecording(bpy.types.Operator):
    """Stop voice recording and start the transcription job, and then execute the request"""
    bl_idname = "voice3d.stop_recording"
    bl_label = "Stop Recording"
    bl_options = {'REGISTER', 'UNDO'}


import openai

# API key must be set up using the OPENAI_API_KEY environment variable!
client = openai.OpenAI()


class StartRecording(bpy.types.Operator):
    """Start voice recording"""
    bl_idname = "voice3d.start_recording"
    bl_label = "Start Recording"
    bl_options = {'REGISTER', 'UNDO'}
    filepath = filepath.name

    def __init__(self):
        self.recording_process = None

    def generate_filename(self, context) -> bool:
        """Check filename availability for the sound file."""

        addon_prefs = context.preferences.addons[__name__].preferences

        timestamp = datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")

        filepath = tempfile.NamedTemporaryFile(prefix=timestamp, suffix=".wav", delete=True)
        filepath.close()
        self.filepath = filepath.name

        if os.path.exists(self.filepath):
            self.report(
                {'ERROR'},
                (
                    f"Could not record audio: ",
                    f"a file already exists where the sound clip would be saved: {self.filepath}",
                ),
            )
            return False

        return True

    def start_recording(self, context) -> bool:
        """Start a process to record audio."""

        assert ffmpeg_exe_path and os_platform in supported_platforms  # poll() should have failed

        addon_prefs = context.preferences.addons[__name__].preferences
        addon_prefs.audio_input_device

        # Set platform dependent arguments.
        if os_platform == 'Linux':
            ffmpeg_command = f'-f alsa -i "{addon_prefs.audio_input_device}"'
        elif os_platform == 'Darwin':
            ffmpeg_command = f'-f avfoundation -i ":{addon_prefs.audio_input_device}"'
        elif os_platform == 'Windows':
            ffmpeg_command = f'-f dshow -i audio="{addon_prefs.audio_input_device}"'

        # This block size command tells ffmpeg to use a small blocksize and save the output to disk ASAP
        file_block_size = "-blocksize 2048 -flush_packets 1"

        # Run the ffmpeg command.
        ffmpeg_command += f' {file_block_size} "{self.filepath}"'
        args = [ffmpeg_exe_path] + shlex.split(ffmpeg_command)
        self.recording_process = subprocess.Popen(args)

        logger.debug("PushToTalk: Started audio recording process")
        logger.debug(f"PushToTalk: {ffmpeg_exe_path} {ffmpeg_command}")
        return True

    def invoke(self, context, event):
        """Called when this operator is starting."""
        # If this operator is already running modal, this second invocation is
        # the toggle to stop it. Set a variable that the first modal operator
        # instance will listen to in order to terminate.
        if context.scene.voice3d_is_recording:
            context.scene.voice3d_should_stop = True
            return {'FINISHED'}

        context.scene.voice3d_is_recording = True

        # Generate the name to save the audio file.
        if not self.generate_filename(context):
            context.scene.voice3d_is_recording = False
            return {'CANCELLED'}

        if not self.start_recording(context):
            context.scene.voice3d_is_recording = False
            return {'CANCELLED'}

        # Start this operator as modal
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        """Periodic update to draw and check if this operator should stop."""

        # Cancel. Delete the current recording.
        if event.type in {'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}

        # Confirm. Create a strip with the current recording.
        if event.type in {'RET'}:
            return self.execute(context)

        # Periodic update
        if event.type == 'TIMER':
            # Listen for signal to stop
            if context.scene.voice3d_should_stop:
                return self.execute(context)
        # Don't consume the input, otherwise it is impossible to click the stop button.
        return {'PASS_THROUGH'}

    def on_cancel_or_finish(self, context):
        """Called when this operator is finishing (confirm) or got canceled."""

        # Unregister from the periodic modal calls.
        if self._timer:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)

        # Finish the sound recording process.
        if self.recording_process:
            self.recording_process.terminate()
            # The maximum amount of time for us to wait for ffmpeg to shutdown in seconds
            maximum_shutdown_wait_time = 3
            try:
                # Wait for ffmpeg to exit until we try to read the saved audio file.
                self.recording_process.wait(maximum_shutdown_wait_time)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "ffmpeg did not gracefully shutdown within "
                    f"{maximum_shutdown_wait_time} seconds."
                )

        # Update this operator's state.
        context.scene.voice3d_is_recording = False
        context.scene.voice3d_should_stop = False

        self.process_audio(self.filepath, context)

    def execute(self, context):
        # Cleanup execution state
        self.on_cancel_or_finish(context)
        return {'FINISHED'}

    def cancel(self, context):
        """Cleanup temporary state if canceling during modal execution."""
        # Cleanup execution state
        self.on_cancel_or_finish(context)
        return {'CANCELLED'}

    def process_audio(self, file_path, context):
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
            self.report({'INFO'}, f"I've understood \"{transcript.text}\"")
        Path(file_path).unlink(missing_ok=True)
        print(context.scene.gpt4_chat_history.items())
        blender_code = generate_blender_code(transcript.text, context.scene.gpt4_chat_history, context, system_prompt)

        message = context.scene.gpt4_chat_history.add()
        message.type = 'user'
        message.content = transcript.text

        # Clear the chat input field
        # context.scene.gpt4_chat_input = ""

        if blender_code:
            message = context.scene.gpt4_chat_history.add()
            message.type = 'assistant'
            message.content = blender_code

            global_namespace = globals().copy()

            try:
                exec(blender_code, global_namespace)
            except Exception as e:
                self.report({'ERROR'}, f"Error executing generated code: {e}")
                # context.scene.gpt4_button_pressed = False
                return {'CANCELLED'}


def generate_blender_code(prompt, chat_history, context, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    for message in chat_history[-10:]:
        if not message.type and not message.content:
            continue
        logger.debug(str({"type": message.type, "content": message.content}))
        if message.type == "assistant":
            messages.append({"role": "assistant", "content": "```\n" + message.content + "\n```"})
        else:
            messages.append({"role": message.type.lower(), "content": message.content})

    # Add the current user message
    messages.append({"role": "user",
                     "content": "Can you please write Blender code for me that accomplishes the following task: " + prompt + "? \n. Do not respond with anything that is not Python code. Do not provide explanations"})

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=False,
        max_tokens=1500,
    )
    print(response)
    completion_text = re.findall(r'```(.*?)```', response.choices[0].message.content, re.DOTALL)[0]
    completion_text = re.sub(r'^python', '', completion_text, flags=re.MULTILINE)
    return completion_text

    # TODO: This was an attempt for streamed responses. This must be completed
    try:
        collected_events = []
        completion_text = ''
        # iterate through the stream of events
        for event in response:
            if event.choices[0].delta.role:
                # skip
                continue
            if len(event.choices[0].delta.dict()) == 0:
                # skip
                continue
            collected_events.append(event)  # save the event response
            event_text = event.choices[0].delta.content
            completion_text += event_text  # append the text
            print(completion_text, flush=True, end='\r')
        completion_text = re.findall(r'```(.*?)```', completion_text, re.DOTALL)[0]
        completion_text = re.sub(r'^python', '', completion_text, flags=re.MULTILINE)

        return completion_text
    except IndexError:
        return None


def menu_func(self, context):
    self.layout.operator(StartRecording.bl_idname)


classes = (VoiceAssistantPanel,
           SEQUENCER_PushToTalk_Preferences
           , StartRecording
           , StopRecording)


def register():
    if os_platform not in supported_platforms:
        logger.warning(
            f"PushToTalk add-on is not supported on {os_platform}. Recording will not work."
        )
    if not ffmpeg_exe_path:
        logger.warning(
            f"PushToTalk add-on could not find ffmpeg separately installed. Recording will not work."
        )

    bpy.types.Scene.voice3d_is_recording = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.voice3d_should_stop = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.gpt4_chat_history = bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)
    bpy.types.PropertyGroup.type = bpy.props.StringProperty()
    bpy.types.PropertyGroup.content = bpy.props.StringProperty()

    # Register as normal.
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.VIEW3D_MT_object.append(menu_func)

    # Sync system detected audio devices with the saved preferences
    addon_prefs = bpy.context.preferences.addons["voicebased3dmodelling"].preferences
    # prop_rna = addon_prefs.rna_type.properties['audio_input_device']
    audio_input_devices = {
        'Linux': addon_prefs.audio_device_linux,
        'Darwin': addon_prefs.audio_device_darwin,
        'Windows': addon_prefs.audio_device_windows,
    }
    saved_setting_value = audio_input_devices[os_platform]

    audio_devices_found = populate_enum_items_for_sound_devices(None, bpy.context)
    assert audio_devices_found  # Should always have an option also when no device is found.

    if saved_setting_value in audio_devices_found:
        # Set the runtime setting to the user setting.
        addon_prefs.audio_input_device = saved_setting_value
    else:
        # Set the runtime setting to the first audio device.
        # This will also update the user setting via the enum's update function.
        addon_prefs.audio_input_device = audio_devices_found[0][0]
        # Log if the user setting got lost.
        if saved_setting_value != "setting not synced yet":
            logger.info(
                f"Could not restore audio device user preference:"
                f"'{saved_setting_value}'. This can happen if the preferred audio device"
                f"is not currently connected."
            )


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.voice3d_is_recording
    del bpy.types.Scene.voice3d_should_stop
    del bpy.types.Scene.gpt4_chat_history
    bpy.types.VIEW3D_MT_object.remove(menu_func)


if __name__ == "__main__":
    register()
