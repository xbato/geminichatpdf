import subprocess

try:
    # Intenta ejecutar 'pdftotext' y obtener la versión
    result = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True, check=True)
    print("pdftotext output:", result.stdout)
    print("pdftotext error output:", result.stderr)  # Imprime la salida de error estándar
except subprocess.CalledProcessError as e:
    print("Error al ejecutar pdftotext:", e)
