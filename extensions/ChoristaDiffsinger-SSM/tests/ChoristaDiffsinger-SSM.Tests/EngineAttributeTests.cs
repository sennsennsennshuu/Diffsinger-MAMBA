using System.Linq;
using System.Reflection;
using TuneLab.Extensions.Voices;
using Xunit;

namespace DiffsingerSSM.Tests;

/// <summary>
/// VoiceEngine attribute is what TuneLab uses to discover the engine type at load time.
/// If this attribute disappears or its value mutates, every existing project that selected
/// this engine breaks silently — there is no migration path. So we lock the value down.
/// </summary>
public class EngineAttributeTests
{
    private static System.Type EngineType =>
        typeof(DiffsingerSSMEngine);

    [Fact]
    public void Engine_Has_VoiceEngineAttribute()
    {
        var attr = EngineType.GetCustomAttribute<VoiceEngineAttribute>(inherit: false);
        Assert.NotNull(attr);
    }

    [Fact]
    public void Engine_VoiceEngineAttribute_Type_Is_DiffsingerSSM()
    {
        var attr = EngineType.GetCustomAttribute<VoiceEngineAttribute>(inherit: false);
        Assert.NotNull(attr);
        Assert.Equal("Diffsinger-SSM", attr!.Type);
    }

    [Fact]
    public void Engine_Implements_IVoiceEngine()
    {
        Assert.True(typeof(IVoiceEngine).IsAssignableFrom(EngineType));
    }

    [Fact]
    public void Engine_Has_Public_Parameterless_Constructor()
    {
        // TuneLab activates extensions via Activator.CreateInstance — a parameterless ctor is mandatory.
        var ctor = EngineType.GetConstructor(System.Type.EmptyTypes);
        Assert.NotNull(ctor);
    }
}