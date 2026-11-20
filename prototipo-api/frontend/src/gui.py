import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_client import APIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DiarizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Whisper Diarizer Interface")
        self.geometry("550x600")
        self.configure(bg="#f5f5f5")
        
        # Initialize API client
        self.api_client = APIClient()
        
        # Application state
        self.selected_file = None
        self.transcript_content = None
        self.is_processing = False
        
        # Configure ttk style
        self.setup_style()
        
        # Create UI elements
        self.create_widgets()
        
        # Check API connection on startup
        self.check_api_connection()
    
    def setup_style(self):
        """Configure the ttk styling for a modern look"""
        style = ttk.Style(self)
        style.theme_use('clam')
        
        # Frame styling
        style.configure("TFrame", background="#ffffff")  
        
        # Label styling
        style.configure("TLabel",
                        background="#ffffff",
                        foreground="#333333",
                        font=("Helvetica", 12))
        style.configure("Title.TLabel",
                        font=("Helvetica", 18, "bold"),
                        background="#ffffff",
                        foreground="#333333")
        style.configure("Status.TLabel",
                        font=("Helvetica", 10),
                        background="#ffffff",
                        foreground="#666666")
        
        # Button styling
        style.configure("TButton",
                        font=("Helvetica", 12),
                        padding=6)
        style.map("TButton", background=[("active", "#e1e1e1")])
        
        # Primary button
        style.configure("Primary.TButton",
                       background="#4CAF50",
                       foreground="#ffffff")
        style.map("Primary.TButton", 
                 background=[("active", "#45a049")])

    def create_widgets(self):
        """Create all GUI elements"""
        # Main container frame
        self.main_frame = ttk.Frame(self, padding="20", style="TFrame")
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Status bar at bottom
        self.status_frame = ttk.Frame(self, style="TFrame")
        self.status_frame.pack(fill="x", side="bottom")
        self.status_label = ttk.Label(
            self.status_frame, 
            text="Checking API connection...", 
            style="Status.TLabel",
            anchor="e"
        )
        self.status_label.pack(padx=10, pady=5, side="right")

        # App title
        title_label = ttk.Label(
            self.main_frame,
            text="Whisper Transcriber & Diarizer",
            style="Title.TLabel",
            anchor="center"
        )
        title_label.pack(pady=(0, 20), fill="x")

        # Create tabbed interface
        self.tab_control = ttk.Notebook(self.main_frame)
        self.tab_control.pack(expand=True, fill="both")
        
        # Transcription tab
        self.transcribe_tab = ttk.Frame(self.tab_control, style="TFrame")
        self.tab_control.add(self.transcribe_tab, text="Transcribe")
        
        # Settings tab
        self.settings_tab = ttk.Frame(self.tab_control, style="TFrame")
        self.tab_control.add(self.settings_tab, text="Settings")
        
        # Create transcription tab content
        self.create_transcribe_tab()
        
        # Create settings tab content
        self.create_settings_tab()

    def create_transcribe_tab(self):
        """Create content for the transcription tab"""
        # File selection section
        file_frame = ttk.Frame(self.transcribe_tab, style="TFrame")
        file_frame.pack(fill="x", pady=10)
        
        file_label = ttk.Label(file_frame, text="Audio File:", style="TLabel")
        file_label.pack(side="left", padx=(0, 10))
        
        self.file_path_var = tk.StringVar(value="No file selected")
        file_path_label = ttk.Label(
            file_frame, 
            textvariable=self.file_path_var,
            wraplength=300
        )
        file_path_label.pack(side="left", fill="x", expand=True)
        
        browse_button = ttk.Button(
            file_frame, 
            text="Browse...", 
            command=self.upload_file
        )
        browse_button.pack(side="right", padx=(10, 0))
        
        # Options frame
        options_frame = ttk.Frame(self.transcribe_tab, style="TFrame")
        options_frame.pack(fill="x", pady=20)
        
        # Left column - Language selection
        language_frame = ttk.Frame(options_frame, style="TFrame")
        language_frame.pack(side="left", fill="both", expand=True)
        
        language_label = ttk.Label(language_frame, text="Language:", style="TLabel")
        language_label.pack(anchor="w", pady=(0, 5))
        
        self.language_var = tk.StringVar(value="Detect Automatically")
        language_options = ["Detect Automatically", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Chinese", "Japanese"]
        language_dropdown = ttk.Combobox(
            language_frame, 
            textvariable=self.language_var,
            values=language_options,
            state="readonly"
        )
        language_dropdown.pack(fill="x", pady=(0, 10))
        
        # Right column - Model selection
        model_frame = ttk.Frame(options_frame, style="TFrame")
        model_frame.pack(side="right", fill="both", expand=True, padx=(20, 0))
        
        model_label = ttk.Label(model_frame, text="Model Size:", style="TLabel")
        model_label.pack(anchor="w", pady=(0, 5))
        
        self.model_var = tk.StringVar(value="tiny")
        model_options = ["tiny", "base", "small", "medium", "large"]
        
        model_dropdown = ttk.Combobox(
            model_frame, 
            textvariable=self.model_var,
            values=model_options,
            state="readonly"
        )
        model_dropdown.pack(fill="x", pady=(0, 10))
        
        # Action buttons
        buttons_frame = ttk.Frame(self.transcribe_tab, style="TFrame")
        buttons_frame.pack(fill="x", pady=10)
        
        self.process_button = ttk.Button(
            buttons_frame, 
            text="Transcribe Audio", 
            command=self.start_processing,
            style="Primary.TButton"
        )
        self.process_button.pack(pady=10)
        
        # Progress indicator
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.transcribe_tab, 
            variable=self.progress_var,
            mode="indeterminate"
        )
        self.progress_bar.pack(fill="x", pady=10)
        
        self.progress_label = ttk.Label(
            self.transcribe_tab, 
            text="", 
            anchor="center"
        )
        self.progress_label.pack(pady=5)
        
        # Transcript display
        transcript_label = ttk.Label(self.transcribe_tab, text="Transcript:", style="TLabel")
        transcript_label.pack(anchor="w", pady=(20, 5))
        
        transcript_frame = ttk.Frame(self.transcribe_tab, style="TFrame")
        transcript_frame.pack(fill="both", expand=True)
        
        self.transcript_text = tk.Text(
            transcript_frame,
            wrap="word",
            height=10,
            width=50,
            font=("Courier", 11)
        )
        self.transcript_text.pack(side="left", fill="both", expand=True)
        
        transcript_scrollbar = ttk.Scrollbar(
            transcript_frame,
            orient="vertical",
            command=self.transcript_text.yview
        )
        transcript_scrollbar.pack(side="right", fill="y")
        self.transcript_text.config(yscrollcommand=transcript_scrollbar.set)
        self.transcript_text.insert("1.0", "Transcribed text will appear here...")
        self.transcript_text.config(state="disabled")
        
        # Download button
        self.download_button = ttk.Button(
            self.transcribe_tab, 
            text="Download Transcript", 
            command=self.download_file, 
            state=tk.DISABLED
        )
        self.download_button.pack(pady=10)

    def create_settings_tab(self):
        """Create content for the settings tab"""
        settings_label = ttk.Label(
            self.settings_tab,
            text="API Connection",
            style="Title.TLabel"
        )
        settings_label.pack(pady=(0, 20))
        
        # API URL setting
        api_frame = ttk.Frame(self.settings_tab, style="TFrame")
        api_frame.pack(fill="x", pady=10)
        
        api_label = ttk.Label(api_frame, text="API URL:", style="TLabel")
        api_label.pack(side="left", padx=(0, 10))
        
        self.api_url_var = tk.StringVar(value="http://127.0.0.1:8000/api/v1")
        api_entry = ttk.Entry(
            api_frame,
            textvariable=self.api_url_var,
            width=40
        )
        api_entry.pack(side="left", fill="x", expand=True)
        
        api_button = ttk.Button(
            api_frame,
            text="Test Connection",
            command=self.check_api_connection
        )
        api_button.pack(side="right", padx=(10, 0))
        
        # About section
        about_frame = ttk.Frame(self.settings_tab, style="TFrame", padding=10)
        about_frame.pack(fill="both", expand=True, pady=20)
        
        about_text = tk.Text(
            about_frame,
            wrap="word",
            height=10,
            width=50,
            font=("Helvetica", 11),
            bg="#f9f9f9",
            relief="flat"
        )
        about_text.pack(fill="both", expand=True)
        about_text.insert("1.0", 
            "Whisper Transcriber & Diarizer\n\n"
            "This application uses OpenAI's Whisper model to transcribe "
            "audio files and identify speakers in the recording.\n\n"
            "For more information and updates, visit the GitHub repository:\n"
            "https://github.com/PoltorProgrammer/Whisper_Transcriber_Diarizer\n\n"
            "Version: 0.1.0"
        )
        about_text.config(state="disabled")

    def check_api_connection(self):
        """Check if API is accessible"""
        self.status_label.config(text="Checking API connection...")
        
        # Update API client with current URL
        self.api_client.base_url = self.api_url_var.get()
        
        # Run in thread to avoid blocking UI
        threading.Thread(target=self._check_api_connection_thread).start()
    
    def _check_api_connection_thread(self):
        """Thread function for API connection check"""
        try:
            result = self.api_client.check_health()
            if result.get("status") == "healthy":
                self.after(0, lambda: self.status_label.config(
                    text="✓ Connected to API",
                    foreground="green"
                ))
            else:
                self.after(0, lambda: self.status_label.config(
                    text="✗ API Error: " + result.get("message", "Unknown error"),
                    foreground="red"
                ))
        except Exception as e:
            self.after(0, lambda: self.status_label.config(
                text=f"✗ Connection failed: {str(e)}",
                foreground="red"
            ))

    def upload_file(self):
        """Open file dialog to select audio file"""
        file_path = filedialog.askopenfilename(
            title="Select an Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.ogg *.flac *.m4a"), 
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.selected_file = file_path
            filename = Path(file_path).name
            self.file_path_var.set(filename if len(filename) < 40 else f"...{filename[-37:]}")
        else:
            self.file_path_var.set("No file selected")
            self.selected_file = None

    def start_processing(self):
        """Start the transcription process"""
        if not self.selected_file:
            messagebox.showerror("Error", "Please select an audio file first!")
            return
        
        # Update UI
        self.is_processing = True
        self.progress_label.config(text="Uploading audio file to API...")
        self.process_button.config(state=tk.DISABLED)
        self.download_button.config(state=tk.DISABLED)
        self.transcript_text.config(state="normal")
        self.transcript_text.delete("1.0", tk.END)
        self.transcript_text.insert("1.0", "Processing...")
        self.transcript_text.config(state="disabled")
        
        # Start progress animation
        self.progress_bar.start(15)
        
        # Start processing in thread
        threading.Thread(target=self.process_file).start()

    def process_file(self):
        """Thread function to handle file processing"""
        try:
            # Get selected options
            language = self.language_var.get()
            model = self.model_var.get()
            
            # Call API
            transcript = self.api_client.transcribe_audio(
                self.selected_file, 
                language, 
                model
            )
            
            # Store result
            self.transcript_content = transcript
            
            # Update UI on main thread
            self.after(0, self.update_ui_after_processing)
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            # Update UI on main thread
            self.after(0, lambda: self.show_error(f"Processing failed: {str(e)}"))

    def update_ui_after_processing(self):
        """Update UI after processing is complete"""
        # Stop progress
        self.progress_bar.stop()
        self.is_processing = False
        
        # Update UI elements
        self.progress_label.config(text="Transcription complete!")
        self.process_button.config(state=tk.NORMAL)
        
        # Display transcript
        self.transcript_text.config(state="normal")
        self.transcript_text.delete("1.0", tk.END)
        
        if self.transcript_content:
            self.transcript_text.insert("1.0", self.transcript_content)
            self.download_button.config(state=tk.NORMAL)
        else:
            self.transcript_text.insert("1.0", "Error: Failed to get transcript from API.")
            self.download_button.config(state=tk.DISABLED)
        
        self.transcript_text.config(state="disabled")

    def show_error(self, message):
        """Display error message and reset UI"""
        self.progress_bar.stop()
        self.is_processing = False
        self.progress_label.config(text=f"Error: {message}")
        self.process_button.config(state=tk.NORMAL)
        messagebox.showerror("Error", message)

    def download_file(self):
        """Save transcript to file"""
        if not self.transcript_content:
            messagebox.showerror("Error", "No transcript available!")
            return

        # Get file name from original audio
        default_name = ""
        if self.selected_file:
            original_name = Path(self.selected_file).stem
            default_name = f"{original_name}_transcript.txt"

        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Save Transcript As",
            initialfile=default_name
        )
        
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(self.transcript_content)
                messagebox.showinfo("Success", f"Transcript saved to {Path(save_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")


if __name__ == "__main__":
    app = DiarizerApp()
    app.mainloop()
