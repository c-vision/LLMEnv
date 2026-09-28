// Apple Vision OCR shim (macOS-only) -- FASE 5 Parte B, opzionale.
// Espone una C API che usa VNRecognizeTextRequest (framework Vision di Apple)
// per estrarre testo da un'immagine. Zero download di modello: il framework e'
// di sistema. Complementa mtmd per chi vuole OCR macOS a costo-zero.
//
// Uso: compilare con clang + framework Vision/Foundation:
//   clang -fobjc-arc -framework Vision -framework Foundation -fobjc-arc \
//         -dynamiclib -o libllm_vision_apple.dylib llm_vision_apple.m
// Poi P/Invoke dal C# (llm_vision_apple_recognize_text).
//
// Nota: il piano teneva Apple Vision come opzione macOS-only NON prioritaria
// (mtmd resta il percorso cross-platform). Questo shim e' il backend
// IVisionCompleter aggiuntivo per le macchine Apple.

#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <AppKit/AppKit.h>

// Riconosce il testo nell'immagine al path. Ritorna una stringa UTF-8
// heap-allocata (malloc) da liberare con llm_vision_apple_free_string,
// o NULL su errore (immagine illeggibile, nessun testo trovato).
const char* llm_vision_apple_recognize_text(const char* image_path);

void llm_vision_apple_free_string(char* s);

static NSString* recognizeText(NSImage* image) {
    CGImageRef cgImage = [image CGImageForProposedRect:NULL context:NULL hints:NULL];
    if (cgImage == NULL) return nil;

    VNRecognizeTextRequest* request = [[VNRecognizeTextRequest alloc] init];
    request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
    request.usesLanguageCorrection = YES;

    VNImageRequestHandler* handler =
        [[VNImageRequestHandler alloc] initWithCGImage:cgImage options:@{}];
    NSError* error = nil;
    if (![handler performRequests:@[request] error:&error]) {
        return nil;
    }

    NSMutableArray* lines = [NSMutableArray array];
    for (VNRecognizedTextObservation* observation in request.results) {
        VNRecognizedText* candidate = [observation topCandidates:1].firstObject;
        if (candidate.string.length > 0) {
            [lines addObject:candidate.string];
        }
    }
    if (lines.count == 0) return nil;
    return [lines componentsJoinedByString:@"\n"];
}

const char* llm_vision_apple_recognize_text(const char* image_path) {
    if (image_path == NULL) return NULL;
    NSString* path = [NSString stringWithUTF8String:image_path];
    NSImage* image = [[NSImage alloc] initWithContentsOfFile:path];
    if (image == nil) return NULL;

    NSString* text = recognizeText(image);
    if (text == nil || text.length == 0) return NULL;

    const char* utf8 = text.UTF8String;
    char* out = (char*)malloc(strlen(utf8) + 1);
    if (out == NULL) return NULL;
    strcpy(out, utf8);
    return out;
}

void llm_vision_apple_free_string(char* s) {
    if (s != NULL) free(s);
}
